import SwiftUI
import MetalKit
import DygenCore

private struct BlitUniforms {
    var scale: SIMD2<Float>
    var offset: SIMD2<Float>
}

/// Blit shader compiled at runtime (avoids needing Xcode 26's separate Metal
/// toolchain component; the driver compiles MSL just fine). Compute kernels in
/// M5+ follow the same runtime-compile approach.
private let blitShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct BlitUniforms { float2 scale; float2 offset; };
struct VOut { float4 pos [[position]]; float2 uv; };

vertex VOut blit_vertex(uint vid [[vertex_id]], constant BlitUniforms& u [[buffer(0)]]) {
    float2 corners[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
    float2 uvs[4]     = { float2(0,1),  float2(1,1),  float2(0,0),  float2(1,0) };
    VOut o;
    o.pos = float4(corners[vid] * u.scale + u.offset, 0, 1);
    o.uv = uvs[vid];
    return o;
}

fragment float4 blit_fragment(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    return tex.sample(s, in.uv);
}
"""

/// Renders the view node's output texture into the MTKView, fitted to the view
/// with pan/zoom, via the blit shader.
final class CanvasRenderer: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    var texture: MTLTexture?
    var zoom: Float = 1
    var pan: SIMD2<Float> = .zero

    init?(device: MTLDevice) {
        guard let queue = device.makeCommandQueue(),
              let lib = try? device.makeLibrary(source: blitShaderSource, options: nil),
              let vfn = lib.makeFunction(name: "blit_vertex"),
              let ffn = lib.makeFunction(name: "blit_fragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pso = try? device.makeRenderPipelineState(descriptor: desc) else { return nil }
        self.queue = queue
        self.pipeline = pso
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else { return }

        if let tex = texture, view.drawableSize.width > 0, view.drawableSize.height > 0 {
            let vw = Float(view.drawableSize.width), vh = Float(view.drawableSize.height)
            let tw = Float(tex.width), th = Float(tex.height)
            let viewAspect = vw / vh, texAspect = tw / th
            var sx: Float, sy: Float
            if texAspect > viewAspect {       // image wider than view → fit width
                sx = zoom; sy = zoom * (viewAspect / texAspect)
            } else {                          // fit height
                sy = zoom; sx = zoom * (texAspect / viewAspect)
            }
            var u = BlitUniforms(scale: SIMD2(sx, sy), offset: pan)
            enc.setRenderPipelineState(pipeline)
            enc.setVertexBytes(&u, length: MemoryLayout<BlitUniforms>.stride, index: 0)
            enc.setFragmentTexture(tex, index: 0)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }
}

/// MTKView subclass adding scroll-to-zoom / drag-to-pan / pinch.
final class CanvasMTKView: MTKView {
    weak var renderer: CanvasRenderer?
    private var lastDrag: CGPoint = .zero

    override func scrollWheel(with event: NSEvent) {
        guard let r = renderer else { return }
        if event.modifierFlags.contains(.command) || !event.hasPreciseScrollingDeltas {
            let f: Float = event.scrollingDeltaY > 0 ? 1.08 : (event.scrollingDeltaY < 0 ? 1/1.08 : 1)
            r.zoom = min(20, max(0.05, r.zoom * f))
        } else {
            r.pan += SIMD2(Float(event.scrollingDeltaX) / Float(bounds.width) * 2,
                           -Float(event.scrollingDeltaY) / Float(bounds.height) * 2)
        }
        needsDisplay = true
    }
    override func magnify(with event: NSEvent) {
        guard let r = renderer else { return }
        r.zoom = min(20, max(0.05, r.zoom * Float(1 + event.magnification)))
        needsDisplay = true
    }
    override func mouseDown(with event: NSEvent) { lastDrag = event.locationInWindow }
    override func mouseDragged(with event: NSEvent) {
        guard let r = renderer else { return }
        let loc = event.locationInWindow
        r.pan += SIMD2(Float(loc.x - lastDrag.x) / Float(bounds.width) * 2,
                       Float(loc.y - lastDrag.y) / Float(bounds.height) * 2)
        lastDrag = loc
        needsDisplay = true
    }
}

/// SwiftUI host: evaluates the document's view node and shows its texture.
struct CanvasView: NSViewRepresentable {
    @ObservedObject var document: Document
    let gpu: GPUContext
    let executor: Executor

    func makeCoordinator() -> Coordinator { Coordinator(executor: executor) }

    func makeNSView(context: Context) -> CanvasMTKView {
        let view = CanvasMTKView(frame: .zero, device: gpu.device)
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1)
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        view.delegate = context.coordinator.renderer
        view.renderer = context.coordinator.renderer
        context.coordinator.refresh(document: document)
        view.needsDisplay = true
        return view
    }

    func updateNSView(_ view: CanvasMTKView, context: Context) {
        context.coordinator.refresh(document: document)
        view.needsDisplay = true
    }

    final class Coordinator {
        let executor: Executor
        let renderer: CanvasRenderer?
        init(executor: Executor) {
            self.executor = executor
            self.renderer = CanvasRenderer(device: executor.ctx.device)
        }
        func refresh(document: Document) {
            guard let renderer else { return }
            do {
                renderer.texture = try executor.evaluate(document.viewNodeID, graph: document.graph)
            } catch {
                renderer.texture = nil
                AppLog.shared.error("\(error)")
            }
        }
    }
}
