# Dygen — Design Document

> A GPU-accelerated, node-based macOS application for the painterly image-processing
> pipeline currently implemented as the slow Python `dygen.py` / `dg_lib.py`.

Status: **draft for review**. This document describes the target architecture, then
critiques itself and revises the risky decisions. The companion [PLAN.md](PLAN.md)
turns it into verifiable milestones.

---

## 1. Vision & Goals

The current tool decomposes an image into per-color blobs, applies textured "brush"
strokes, recomposites, and post-processes (glow / sharpen). It is correct but runs in
**minutes** because every pixel op is an interpreted Python double-loop executed once
per palette color (~400M iterations for a 1024² image with 64 colors).

We are rebuilding it as a native macOS app with three goals:

1. **Fast** — move the pixel work to the GPU. Target: full pipeline in well under a
   second, fast enough that parameter changes feel live.
2. **Interactive & node-based** — express the pipeline as an editable node graph
   (Read → ops → Write), with a Canvas to see results, Properties to tune parameters,
   and a Log for debugging. The README's whole pitch is *"start messing with
   parameters"*; the app should make that a real-time loop, not an edit-a-constant /
   wait / open-a-PNG loop.
3. **A reusable foundation** — the docking/window system, node editor, properties, and
   log are general-purpose. We extract them into a **WindowKit** layer that this app
   (and future ones) builds on. Much of this is lifted and de-coupled from the existing
   `dray` app, which already has production-quality versions of all four.

### Non-goals (v1)

- **Not pixel-identical** to the Python output. Generative art; we validate by eye.
  This frees us to use Apple's optimized filters and a different RNG.
- **Not a general image editor.** Focused on this painterly pipeline plus enough
  node-graph generality to experiment.
- **Single document at a time.** No MDI, no tabs-of-documents, no animation/timeline,
  no plugin system. All deferred.

---

## 2. Architectural Overview

Five layers, each depending only on those below it:

```
┌──────────────────────────────────────────────────────────────┐
│  App Shell  (DygenApp)                                         │
│  windows wiring · menus · document lifecycle · selection      │
├──────────────────────────────────────────────────────────────┤
│  Editors                                                       │
│  CanvasView · NodeEditor · PropertyInspector · LogView        │
├───────────────┬──────────────────────────────────────────────┤
│  WindowKit    │  Document / Model                             │
│  (docking,    │  Document · Graph · Node · Connection ·       │
│   layout      │  NodeValue · NodeRegistry · CommandBus        │
│   persistence)│                                               │
│               ├──────────────────────────────────────────────┤
│               │  Execution Engine                             │
│               │  DAG eval · dirty tracking · per-node cache   │
│               ├──────────────────────────────────────────────┤
│               │  GPU Backend                                  │
│               │  Core Image + custom Metal kernels + MPS      │
└───────────────┴──────────────────────────────────────────────┘
```

**Data flow (the core loop):**

```
user edits a parameter / wires a connection
        │
        ▼
  Command.perform()  ──►  mutates Document.graph  ──►  returns inverse (for undo)
        │
        ▼
  ChangeBus.didChange(nodeID)
        │
        ├──►  Editors refresh (SwiftUI/AppKit observe the change)
        │
        └──►  Executor marks node + downstream dirty
                   │
                   ▼
              Canvas requests the "view" node's image
                   │
                   ▼
              Executor lazily re-evaluates only dirty nodes (cached otherwise)
                   │
                   ▼
              GPU renders at preview resolution  ──►  Canvas displays
```

The **Document is the single source of truth.** Nothing else holds authoritative state.
This mirrors dray's most important pattern and is the reason undo, save/load, and live
re-evaluation all stay coherent.

---

## 3. WindowKit — the docking foundation

Lifted from dray's `LayoutManager` / `LayoutNode` / `AreaView` / `LayoutDivider`, which
are a Blender-style binary-split docking system with **near-zero domain coupling**.

**Core model (a recursive, Codable split tree):**

```swift
public indirect enum LayoutNode: Identifiable, Codable, Equatable {
    case area(AreaState)     // a leaf: shows one editor (with tabs)
    case split(SplitState)   // an internal node: children + axis + fractions
}

public struct AreaState:  { var tabs: [WindowKind]; var activeTab: Int }
public struct SplitState: { var axis: Axis; var children: [LayoutNode]; var fractions: [CGFloat] }
public struct Workspace:  Codable { var name: String; var root: LayoutNode }
```

**How a leaf decides what to show** — dependency injection through the SwiftUI
environment, so WindowKit never imports anything app-specific:

```swift
// WindowKit declares the slot; the app fills it in.
public typealias WindowContentBuilder = (WindowKind, UUID) -> AnyView
// AreaView calls:  contentBuilder(area.activeTab, area.id)
```

`WindowKind` is the one thing the app supplies (dray calls it `EditorType`). For us:

```swift
enum WindowKind: String, Codable, CaseIterable { case canvas, nodeEditor, properties, log }
```

**What WindowKit owns:** the split tree, `LayoutManager` (mutation + JSON persistence to
`~/Library/Application Support/Dygen/layout.json`), the recursive `AreaView` /
`LayoutNodeView` renderer, `LayoutDivider` (proportional drag-resize with min-pane
clamping), and workspace switching. **What stays in the app:** `WindowKind` and the
content builder that maps a kind to a concrete editor view.

WindowKit is a **standalone Swift package** with no dependency on the Document model — it
is pure UI shell. (Verified extraction effort from dray: small; the types are already
factored for reuse.)

---

## 4. Document & Data Model

The model is **primary and executable** — unlike dray, where the graph is derived from a
USD layer. There is no USD here; the graph *is* the document.

```swift
final class Document: ObservableObject {
    var graph: Graph
    var selection: Set<Node.ID>
    var viewNodeID: Node.ID?     // which node's output the Canvas shows
    let commands = CommandBus()
    let changes  = ChangeBus()
    var fileURL: URL?
}

struct Graph: Codable {
    var nodes: [Node]
    var connections: [Connection]
}

struct Node: Identifiable, Codable {
    let id: UUID
    var type: String              // registry key, e.g. "read", "quantize", "paint"
    var position: CGPoint         // node-editor canvas position
    var params: [String: NodeValue]
}

struct Connection: Codable {      // output port → input port
    var fromNode: UUID; var fromPort: String
    var toNode:   UUID; var toPort:   String
}

enum NodeValue: Codable, Equatable {     // typed parameter values (cf. dray MaterialValue)
    case float(Float), int(Int), bool(Bool)
    case color(Float, Float, Float)
    case float3(Float, Float, Float)
    case string(String)           // also used for file paths / enum choices
}
```

### Node type registry

Each node *type* is described once by a `NodeDescriptor`. This single description drives
(a) the ports the node editor draws, (b) the parameter rows the inspector renders, and
(c) which GPU operation the executor runs. This is the schema-driven approach from dray's
`PropertyTemplate` / `SchemaRegistry`, unified with execution.

```swift
struct PortSpec   { let name: String; let kind: PortKind }   // v1: PortKind = .image only
struct ParamSpec  { let key: String; let label: String; let ui: ParamUI; let `default`: NodeValue
                    let range: ClosedRange<Float>?; let choices: [String]? }
enum ParamUI      { case scrubFloat, scrubInt, toggle, color, filePicker, dropdown }

struct NodeDescriptor {
    let type: String
    let category: String                 // "IO", "Quantize", "Paint", "Filter"…
    let inputs:  [PortSpec]
    let outputs: [PortSpec]
    let params:  [ParamSpec]
    let make: () -> NodeOp               // factory for the GPU operation
}

enum NodeRegistry { static var all: [String: NodeDescriptor] }   // populated at launch
```

### Serialization

Plain **JSON via `Codable`** (we own the format — no reason for anything heavier).
Versioned top-level field for forward migration. A `.dygen` file is the encoded `Graph`
plus document metadata (view node, canvas state). Human-readable, diff-able, git-friendly.

```jsonc
{
  "version": 1,
  "viewNode": "…uuid…",
  "graph": {
    "nodes": [
      { "id": "…", "type": "read",  "position": [40,120], "params": { "path": {"string": "ref/ref.png"} } },
      { "id": "…", "type": "quantize", "position": [260,120],
        "params": { "colors": {"int": 64}, "kmeans": {"int": 1} } }
    ],
    "connections": [ { "fromNode": "…", "fromPort": "out", "toNode": "…", "toPort": "in" } ]
  }
}
```

### Commands & undo

Straight from dray: a `Command` mutates the document and **returns its own inverse**; the
`CommandBus` keeps undo/redo stacks (no whole-document snapshots). `.merge` behavior
coalesces continuous edits (a slider drag = one undo step). Every model mutation goes
through a command so undo, save-dirty tracking, and the change bus stay consistent.

```swift
protocol Command { var label: String { get }
                   func perform(_ doc: Document) -> Command }   // returns inverse
// e.g. SetParam, AddNode, DeleteNode, Connect, Disconnect, MoveNode
```

---

## 5. Execution Engine

The graph is a DAG. Each node is a `NodeOp` that takes input textures + params and
encodes GPU work producing an output texture. Evaluation is **cached and dirty-driven** so
editing only re-runs what changed.

```swift
protocol NodeOp {
    // Encodes its compute dispatch(es) into `enc`, returns the texture it wrote.
    func evaluate(inputs: [String: MTLTexture], params: [String: NodeValue],
                  enc: MTLComputeCommandEncoder, ctx: GPUContext) -> MTLTexture
}
```

- **Currency between nodes is `MTLTexture`** (raw Metal — see §6). Unlike Core Image's
  deferred recipes, textures are concrete pixels, so the executor must manage caching and
  texture lifetime itself (a `MTLHeap`/texture pool keyed by size+format avoids per-eval
  allocation churn).
- **Caching:** each node memoizes its output `MTLTexture` keyed by a hash of (its params +
  upstream output hashes). A parameter change invalidates that node and everything
  downstream; untouched branches keep their cached textures.
- **Dirty propagation:** `ChangeBus.didChange(nodeID)` → executor marks the node and its
  transitive successors dirty. The Canvas pulls the view node; only dirty nodes re-run,
  encoded into one `MTLCommandBuffer` per evaluation.
- **Topology:** topological sort with cycle rejection (the node editor also refuses to
  create a cycle at connect time).

### Node granularity — the central design decision

The Python pipeline's defining feature is a **loop over palette colors** (extract a mask
per color, brush it, composite in order). You cannot make 64 colors into 64 graph nodes,
and you should not collapse the whole thing into one opaque "Dygen" node either.

**Resolution: medium-grained, per-*stage* nodes.** Each node corresponds to one pipeline
*stage* and is a self-contained GPU operation that may *internally* iterate over the
palette. The graph expresses the legible sequence of stages; the per-color loop is an
implementation detail inside the relevant nodes' shaders. This matches how compositors
(Nuke, Blender) are structured and keeps the graph readable while staying faithful.

So `PaintBlobs`, `BrushStrokes`, and `CompOriginal` are single nodes that each loop over
the palette internally (as cheap GPU passes), while `Read`, `Quantize`, `Cleanup`,
`Pixelate`, `Glow`, `Sharpen`, `Write` are straightforward one-shot nodes.

---

## 6. GPU Backend

**Decision: raw Metal compute.** Each node is one or more hand-written `.metal` compute
kernels dispatching one thread per pixel; `MTLTexture` is the currency between nodes. We
hand-manage the command queue, encoders, and texture pool. **MetalPerformanceShaders
(MPS)** supplies the optimized standard filters (Gaussian blur, convolution) so we don't
re-derive separable convolutions. Rationale: maximum control over the pipeline and memory,
a single uniform GPU paradigm (no CIImage/MTLTexture bridging), and the ops already map
cleanly to per-pixel compute kernels.

**The index-image trick is central here.** Quantize produces an **index texture**
(`R8Uint`, 0..N-1) + a **palette buffer** (`MTLBuffer` of RGB). The painterly kernels treat
"the mask for color *i*" as `index == i` tested in-shader rather than materializing N mask
textures. Pass-1 blobs collapse to a single kernel (`palette[index] + jitter`); the
ordered brush/comp passes are **N sequential compute dispatches** into a ping-ponged canvas
texture (each dispatch is microseconds at 1MP, so 64 of them is still sub-millisecond).

**Texture management:** a size+format-keyed texture pool (or `MTLHeap`) hands out transient
textures and recycles them, so re-evaluation doesn't allocate. Per-node cached outputs are
retained until invalidated by dirty propagation (§5).

**Quantization** (median-cut / k-means) resists the GPU and runs once; it stays on the CPU
(Accelerate/vImage or a small k-means), writing the index texture + palette buffer the
kernels consume.

**Display & export:** the Canvas is an `MTKView` whose drawable is blitted/drawn from the
view node's output texture; export reads the full-res output texture back and encodes a PNG
(`MTLTexture` → `CGImage` → PNG, or vImage).

**Determinism:** the brush jitter uses a seedable in-shader hash RNG (`hash(pixel,
strokeIndex, seed)`), so a given seed reproduces the same art (critique C6).

### Op → kernel mapping

| DYImage op              | Node            | Implementation                                       |
|-------------------------|-----------------|------------------------------------------------------|
| load / save             | Read / Write    | image load → texture / texture → PNG (CGImage/vImage)|
| quantize                | Quantize        | CPU k-means → index texture + palette buffer          |
| cleanup (ModeFilter)    | Cleanup         | compute kernel: mode of indices over window          |
| pixilated               | Pixelate        | compute kernel: sample at floor(coord/n)             |
| blur                    | Blur            | `MPSImageGaussianBlur`                                |
| clamp/mult/blend/multed | (params on nodes / small kernels) | trivial per-pixel compute kernels  |
| paint (blobs)           | PaintBlobs      | single kernel: palette[index] + seeded jitter         |
| paint (brush, ordered)  | BrushStrokes    | N sequential kernel passes (mask→blur→clamp→stencil→gate) |
| comp                    | CompOriginal    | N sequential kernel passes (masked copy)             |
| glow / glow_blur        | Glow            | threshold kernel → `MPSImageGaussianBlur` → add kernel |
| sharpen                 | Sharpen         | `MPSImageConvolution` (3×3 sharpen)                  |

---

## 7. Editors (the windows)

| Window         | Source            | Approach                                                                 |
|----------------|-------------------|-------------------------------------------------------------------------|
| **Canvas**     | new               | `MTKView` drawing the view node's output `MTLTexture` directly; pan/zoom; preview-res live, full-res on export. |
| **Node Editor**| dray (port now)   | Port dray's AppKit `NSView` + CoreGraphics `NodeGraphCanvasView` up front: nodes, bezier wires, drag/connect/box-select/cut-stroke-delete/trackpad zoom/framing. Strip USD/ShaderRegistry; drive ports from `NodeDescriptor`. |
| **Properties** | dray (near copy)  | `PropertyTemplate`-style rows from the selected node's `ParamSpec`s; `ScrubField` for numbers; writes back via `SetParam` command. |
| **Log**        | dray (copy)       | `AppLog` ring buffer + `LogView`; both already domain-agnostic.         |

Selection is owned by the `Document`; Properties and Canvas observe it. Choosing a node's
output as the Canvas view is an explicit action ("view this node", like a compositor's
viewer pin).

---

## 8. Critique & Revisions

A candid pass over the design above, with the changes each critique produces.

**C1 — Raw Metal means we own caching and texture lifetime.** Unlike Core Image, nothing
defers or pools for us; naive evaluation would allocate textures every frame and thrash.
*Revision:* the executor owns a **size+format-keyed texture pool** (or `MTLHeap`) for
transients, retains per-node cached outputs until dirty, and encodes a whole evaluation
into one command buffer. Color is treated as raw bytes in the textures' own space (we don't
apply working-space conversions), which keeps the math close to the Python and suits visual
equivalence. This is more code than Core Image but is exactly the control we chose.

**C2 — The node editor is ~1500 lines coupled to USD/ShaderRegistry.** Porting it is the
biggest single chunk of work. *Decision (chosen):* port dray's AppKit `NodeGraphCanvasView`
**up front** (M3), retyped off `NodeDescriptor` with USD/ShaderRegistry stripped, for the
richer interactions (cut-stroke delete, box-select, LOD, trackpad zoom/framing) from day
one. *Mitigation for the risk this front-loads:* M0–M2 (shell, model, persistence, log)
do not depend on the node editor, so they proceed in parallel and the port has a working
`Document`/`NodeDescriptor` to bind against when it lands.

**C3 — Palette has to flow between nodes, but ports are image-only.** `Quantize` produces
a palette that `PaintBlobs`/`BrushStrokes` need. Adding a typed "palette" port early
complicates the type system. *Revision:* keep **image-only ports** in v1. The painterly
nodes take the *quantized image* as input and recompute the palette internally (cheap:
one histogram). No second port type until we actually need value-wires. This is the
single biggest simplification in the design.

**C4 — Granularity could still feel "not nodey."** Encapsulating the palette loop inside
`BrushStrokes` means users can't rewire individual strokes. *Revision:* accepted on
purpose — strokes are not a user concern; *stages* are. We expose the knobs (stencils,
amounts, counts) as parameters. If demand appears, a future "iterator" meta-node can
generalize the loop, but it is explicitly out of scope for v1.

**C5 — Scope is large; risk of a long stretch with nothing runnable.** *Revision:* the
plan is sliced so **every milestone ends in something you can build and see** (M0 blank
window → M1 docking → … → M5 Read→Write image on screen → M7 full pipeline). Vertical
slices over horizontal layers. WindowKit, Document, and Executor each land with a visible
proof.

**C6 — Determinism / the brush RNG.** Python uses `random.seed(2)` + per-stroke jitter.
*Revision:* implement a seedable hash-based RNG in-shader (e.g. PCG/`hash(pixel, strokeID,
seed)`). Same seed → reproducible art across runs; exact match to Python is a non-goal.

**C7 — Re-evaluating on every slider tick could stutter.** With raw Metal there is no free
deferral, so this needs real care. *Revision:* (a) per-node texture cache means only the
dirty sub-DAG re-encodes; (b) the Canvas renders at *preview* resolution during interaction
and full-res only on export/idle; (c) the per-color brush/comp passes are microsecond-scale
on the GPU even at 64 iterations, so a full re-eval is still well within a frame; (d)
`SetParam` uses the command bus's `.merge` so a drag is one undo step. If a drag ever
outpaces the GPU, coalesce to the latest value (drop intermediate frames).

**C8 — Single window-kind enum vs. protocol in WindowKit.** dray uses a concrete
`EditorType` enum; a generic/protocol is "cleaner" but heavier. *Revision:* use a simple
`WindowKind` enum supplied by the app (matches dray, proven, trivially Codable). Generality
via protocol is YAGNI for one app.

**C9 — Build tooling.** No `xcodegen`/`tuist` present; hand-maintaining a `.pbxproj` is
painful. *Revision:* WindowKit and the model/executor live in **Swift Packages** (clean,
testable, CI-friendly, no project file churn); only the thin app shell (app bundle, Metal
resources, entitlements) is an Xcode target that depends on those packages. Most code is
thus in packages we can `swift build`/`swift test` headlessly.

### Net changes from critique

1. Executor owns a texture pool + per-node cache; one command buffer per evaluation.
2. **Image-only ports** in v1; painterly nodes recompute palette internally.
3. Most code in **Swift Packages**; thin Xcode app shell on top.
4. Preview-resolution rendering during interaction; full-res on export.
5. Seedable in-shader RNG for reproducible brush jitter.

These are folded into [PLAN.md](PLAN.md).

---

## 9. Decisions (resolved)

| Decision        | Choice                                                                 |
|-----------------|-----------------------------------------------------------------------|
| **GPU backend** | **Raw Metal compute** — `MTLTexture` currency, hand-written `.metal` kernels, MPS for blur/convolution. |
| **Node editor** | **Port dray's AppKit canvas now** (M3), retyped off `NodeDescriptor`. |
| **dray reuse**  | **Copy & adapt** the four subsystems into this repo's own packages; no external dray dependency. |
| **Fidelity**    | **Visual equivalence** — not pixel-exact; seedable RNG, raw-byte color space. |
