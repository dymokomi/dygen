# Dygen — Implementation Plan

Companion to [DESIGN.md](DESIGN.md). The work is sliced into **vertical milestones**:
every one ends in something you can build and *see*, so we can "check it out" at each
step before moving on. Each milestone lists its **goal**, **work**, and an explicit
**✅ Check** (how you verify it's done).

Resolved decisions (see [DESIGN.md §9](DESIGN.md)):
- **GPU backend:** raw Metal compute — `MTLTexture` currency, hand-written `.metal`
  kernels, MPS for blur/convolution.
- **Node editor:** port dray's AppKit `NodeGraphCanvasView` up front (M3).
- **dray reuse:** copy & adapt the four subsystems into this repo's own packages.
- **Fidelity:** visual equivalence (not pixel-exact).

Guiding rules from the design critique:
- Most code lives in **Swift Packages** (`WindowKit`, `DygenCore`); a thin Xcode app
  target (`Dygen.app`) wires them together and owns the bundle/Metal resources.
- **Image-only ports** in v1; painterly nodes recompute palette internally.
- **Preview-resolution** rendering while interacting; full-res on export.
- The executor owns a **texture pool + per-node cache**; one command buffer per eval.

Layout (as built in M0):

```
dygen/
├─ dygen.py, dg_lib.py          # the reference implementation (kept for validation)
├─ Package.swift                # WindowKit + DygenCore libraries (local Swift package)
├─ Sources/
│  ├─ WindowKit/                # docking/layout  ← from dray
│  ├─ DygenCore/                # Document, Graph, executor, GPU ops
│  └─ Dygen/                    # SwiftUI app sources (built by the xcodeproj target)
├─ Tests/DygenCoreTests/        # logic tests run headlessly via `swift test`
├─ Dygen.xcodeproj             # single app target; consumes the local package
│                              # (XCLocalSwiftPackageReference "."), file-system-
│                              # synchronized source group so new files auto-include
├─ dev.sh                       # build|run|test|clean
└─ docs/                        # DESIGN.md, PLAN.md
```

The `.xcodeproj` is managed with the `pbxproj` Python lib when manual edits are
needed; most source additions need no project edits thanks to the synchronized group.

---

## M0 — Skeleton app builds & runs
**Goal:** an empty SwiftUI macOS app launches to a blank window. Packages exist and link.

**Work**
- Create `Dygen.xcodeproj` (macOS app, SwiftUI lifecycle, min macOS 14).
- Create empty `WindowKit` and `DygenCore` Swift packages; add both as dependencies of
  the app. Print a line from each at launch to prove linkage.

**✅ Check:** `Dygen.app` launches; a blank window appears; console shows the two packages'
init lines. `swift build` succeeds in each package.

---

## M1 — WindowKit: docking layout with placeholder editors
**Goal:** a Blender-style dockable window system; split/resize panes; pick which editor a
pane shows; layout survives relaunch.

**Work**
- Port from dray into `WindowKit`: `LayoutNode`/`AreaState`/`SplitState`/`Workspace`,
  `LayoutManager` (tree mutation + JSON persistence), `AreaView`/`LayoutNodeView`
  recursive renderer, `LayoutDivider` (drag-resize), `AreaHeader` (tab + editor picker).
- Replace dray's `EditorType` with the app's `WindowKind` enum + the
  `WindowContentBuilder` environment slot. Editors are placeholder colored views for now.
- Persist to `~/Library/Application Support/Dygen/layout.json`.

**✅ Check:** Split a pane horizontally/vertically; drag the divider to resize; change a
pane's editor via its header menu; quit and relaunch → the exact layout is restored.

---

## M2 — Document model + scene save/load + Log window
**Goal:** a real `Document` holding a node graph, with JSON save/load and a working Log.

**Work** (in `DygenCore`, pure logic — unit-testable headless)
- `Document`, `Graph`, `Node`, `Connection`, `NodeValue` (Codable).
- `NodeDescriptor` / `ParamSpec` / `PortSpec` + `NodeRegistry`; register two trivial
  types: `Read` and `Write` (no GPU yet — descriptors only).
- `CommandBus` + commands: `AddNode`, `DeleteNode`, `Connect`, `Disconnect`, `MoveNode`,
  `SetParam` (each returns its inverse). `ChangeBus` for notifications.
- `.dygen` save/load (JSON). Document-dirty tracking; standard `File ▸ Open/Save` menus.
- Port dray's `AppLog` + `LogView` into the app; route command/exec logs through it.

**✅ Check:** Unit tests: build a graph in code, save, reload, assert equality; run a
command then undo and assert the document matches the pre-command state. In-app: the Log
window shows command traces; Save/Open round-trips a `.dygen` file.

---

## M3 — Node editor (port dray's AppKit canvas) + Properties
**Goal:** see and edit the graph with dray-grade interactions; select a node and tune its
parameters. This front-loads the hardest port (critique **C2**) — but M0–M2 don't depend
on it, so it binds against a ready `Document`/`NodeDescriptor`.

**Work**
- Port dray's `NodeGraphCanvasView` (AppKit `NSView` + CoreGraphics) into the app:
  node/edge/grid drawing, LOD, bezier wires, `NodeGraphViewport` zoom/pan transforms.
  Strip USD/`SdfPrimSpec`/`ShaderRegistry`; drive nodes/ports from `Node` + `NodeDescriptor`,
  positions from `Node.position`.
- Wire interactions to the command bus: drag nodes (`MoveNode`), port→port connect
  (`Connect`, reject cycles/dupes), box-select, cut-stroke wire delete (`Disconnect`),
  delete key (`DeleteNode`), right-click "Add Node" from the registry (`AddNode`).
- Replace dray's port-type palette/compatibility with our `PortKind` (v1: image-only).
- Properties window: render the selected node's `ParamSpec`s as rows (port dray's
  `ScrubField` for numbers, plus toggle/color/dropdown/filePicker); edits emit `SetParam`.

**✅ Check:** Add `Read` and `Write` from the canvas, wire them, box-select and multi-move,
cut-stroke to delete the wire, undo/redo each; trackpad pan/pinch-zoom and frame-all work;
select `Read`, change its file-path param in Properties and see it persist through
save/load.

---

## M4 — GPU context + Canvas + Read→Write executes
**Goal:** the first **vertical slice through the whole stack** — an image actually flows
through the graph as `MTLTexture`s and appears on screen.

**Work**
- `GPUContext` (`MTLDevice`, command queue, texture pool); `NodeOp` protocol (encodes
  into a compute encoder, returns its output `MTLTexture`).
- Implement `Read` (image file → `MTLTexture`) and `Write` (`MTLTexture` → PNG via
  `CGImage`/vImage) ops.
- Execution engine: topo-sort, dirty propagation, per-node `MTLTexture` cache, one
  `MTLCommandBuffer` per evaluation, transient-texture pooling.
- Canvas window: `MTKView` drawing the **view node's** output texture; pan/zoom;
  "view this node" action; preview-res while interacting.

**✅ Check:** Point `Read` at `ref/ref.png`, set it as the Canvas view → the image shows.
Wire `Read → Write`, trigger Write → a PNG appears on disk identical to the input. Change
the Read path → Canvas updates live.

---

## M5 — First real GPU ops: Blur, Sharpen, Pixelate
**Goal:** prove the op pattern end-to-end with both MPS and custom compute kernels, and
prove live interactivity.

**Work**
- `Blur` (`MPSImageGaussianBlur`) and `Sharpen` (`MPSImageConvolution`, 3×3) — MPS.
- `Pixelate` — first **custom `.metal` compute kernel**, sampling `floor(coord/n)`. Sets up
  the Metal shader build (default library) + dispatch pipeline in the app target.
- Wire into Properties (radius/size sliders) with `.merge` undo coalescing.

**✅ Check:** Build `Read → Pixelate → Blur → Sharpen → Canvas`; drag the Pixelate/Blur
sliders and watch the Canvas update in real time; confirm one drag = one undo step.

---

## M6 — Quantize + Cleanup (the index-image foundation)
**Goal:** the palette/index representation the painterly nodes depend on.

**Work**
- `Quantize`: CPU k-means/median-cut → index image + palette (matches `color_count`,
  `kmeans`). Decide internal carrier (index texture + palette buffer) per DESIGN §6.
- `Cleanup`: custom compute kernel — mode of indices over the window (`cluster` radius).
- Validate index/palette by reconstructing the RGB image and eyeballing vs. Python's
  `*.blobs` intermediate.

**✅ Check:** `Read → Quantize → Cleanup → Canvas` reproduces the flat-color, despeckled
look of Python's early stage; changing `colors`/`cluster` updates live.

---

## M7 — The painterly nodes: PaintBlobs, BrushStrokes, CompOriginal, Glow
**Goal:** reproduce the full `dygen.py` look as a node graph — the headline result.

**Work**
- `PaintBlobs`: single compute kernel, `palette[index] + seeded jitter` (collapses
  Python's pass-1 64-loop to one pass).
- `BrushStrokes`: N sequential compute passes into a ping-ponged canvas — per palette
  color: build soft mask (`index==i` → MPS blur → clamp), multiply a stencil (extra image
  input), composite with the `amount`/`volume_diff` luminance gate. Seedable in-shader RNG
  (critique **C6**).
- `CompOriginal`: N sequential passes copying original-image pixels through textured masks.
- `Glow`: threshold kernel → `MPSImageGaussianBlur` → additive kernel.
- Stencils/brushes come in as extra image inputs (Read nodes pointed at `tex/*.png`).

**✅ Check:** Assemble the full graph; output visually matches `out/ref.painted.v1.png` /
`out/ref.sharpened.v1.png` (side-by-side, eyeball). End-to-end runtime is well under a
second at full res, vs. the Python minutes.

---

## M8 — Polish: presets, export, interactivity
**Goal:** ship-quality loop for the "mess with parameters" workflow.

**Work**
- Ship a default `.dygen` scene reproducing `dygen.py`'s graph (the README example).
- Full-res PNG export with versioned filenames (parity with Python's `.vN.png`).
- Canvas niceties: fit/100%/zoom, background, before/after.
- Performance pass: confirm dirty-sub-DAG caching; preview vs. full-res switching;
  validate slider drags stay smooth.

**✅ Check:** Open the bundled scene, reproduce the reference output via Export, then tune
`colors`/`cluster`/`glow`/stencils with live Canvas feedback — the whole original
workflow, now interactive.

---

## Dependency graph & sequencing

```
M0 ─► M1 ─► M2 ─► M3 ─► M4 ─► M5 ─► M6 ─► M7 ─► M8
                  │
                  └─ (M3 is the big dray AppKit node-editor port; M0–M2 don't depend on it)
```

- **M0–M2** stand up the shell, model, and persistence (no GPU); independent of M3.
- **M3–M4** are the first user-visible vertical slice (edit graph → see image).
- **M5–M7** add raw-Metal GPU ops, culminating in the full painterly result.
- **M8** is polish.

## What "done" means for v1

Open the app, load the default scene, see `ref.png` painted in real time, tweak parameters
with live feedback, and export a full-res PNG — all in well under a second per change,
reproducing the Python tool's look at interactive speed.
