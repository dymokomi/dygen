import Metal
import MetalPerformanceShaders
import Foundation

/// Resolve the clean quantized index for a painterly input (or recompute it
/// from RGB + palette as a fallback).
private func resolveIndex(_ input: NodeImage, ctx: GPUContext) throws -> MTLTexture? {
    guard let src = input.texture, let palette = input.palette, input.paletteCount > 0 else { return nil }
    return try input.indexTexture ?? PaletteUtil.indexTexture(rgb: src, palette: palette, count: input.paletteCount, ctx: ctx)
}

/// Pass 1: recolour each region (palette colour + per-colour jitter).
public struct PaintBlobsOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let input = inputs["in"], let src = input.texture, let palette = input.palette,
              let idx = try resolveIndex(input, ctx: ctx),
              let out = ctx.makeTexture(width: src.width, height: src.height),
              let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return inputs["in"] }
        let pso = try ctx.pipeline("paint_blobs")
        enc.setComputePipelineState(pso)
        enc.setTexture(out, index: 0); enc.setTexture(idx, index: 1)
        enc.setBuffer(palette, offset: 0, index: 0)
        var count = UInt32(input.paletteCount); enc.setBytes(&count, length: 4, index: 1)
        var seed = UInt32(max(0, params["seed"]?.intValue ?? 2)); enc.setBytes(&seed, length: 4, index: 2)
        PaletteUtil.dispatch(enc, pso, src.width, src.height)
        enc.endEncoding(); cb.commit()
        return NodeImage(texture: out, indexTexture: idx, palette: palette, paletteCount: input.paletteCount)
    }
}

/// Pass 2: layered, textured brush strokes (one composite pass per palette colour).
public struct BrushStrokesOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let input = inputs["in"], let src = input.texture, let palette = input.palette,
              let idx = try resolveIndex(input, ctx: ctx) else { return inputs["in"] }
        let w = src.width, h = src.height
        let count = input.paletteCount
        let amount = Float(params["amount"]?.floatValue ?? 0.5)
        let softness = max(0.5, Float(params["softness"]?.floatValue ?? 12))
        let volumeDiff = Float(params["volumeDiff"]?.floatValue ?? 0.03)
        let stencil = inputs["stencil"]?.texture

        guard var canvasA = ctx.makeTexture(width: w, height: h),
              var canvasB = ctx.makeTexture(width: w, height: h),
              let mask = ctx.makeTexture(width: w, height: h),
              let blurred = ctx.makeTexture(width: w, height: h),
              let cb = ctx.queue.makeCommandBuffer() else { return inputs["in"] }

        // canvasA = copy of the incoming image.
        try ctx.encode(kernel: "copy_tex", output: canvasA, inputs: [src], in: cb)

        let maskPSO = try ctx.pipeline("region_mask")
        let compPSO = try ctx.pipeline("brush_composite")
        let blur = MPSImageGaussianBlur(device: ctx.device, sigma: softness)
        blur.edgeMode = .clamp

        for i in 0..<count {
            // mask = (index == i)
            if let enc = cb.makeComputeCommandEncoder() {
                enc.setComputePipelineState(maskPSO)
                enc.setTexture(mask, index: 0); enc.setTexture(idx, index: 1)
                var t = UInt32(i); enc.setBytes(&t, length: 4, index: 0)
                PaletteUtil.dispatch(enc, maskPSO, w, h)
                enc.endEncoding()
            }
            blur.encode(commandBuffer: cb, sourceTexture: mask, destinationTexture: blurred)
            // composite into the other canvas
            if let enc = cb.makeComputeCommandEncoder() {
                enc.setComputePipelineState(compPSO)
                enc.setTexture(canvasB, index: 0); enc.setTexture(canvasA, index: 1)
                enc.setTexture(blurred, index: 2)
                enc.setTexture(stencil ?? blurred, index: 3)
                enc.setBuffer(palette, offset: 0, index: 0)
                var ci = UInt32(i); enc.setBytes(&ci, length: 4, index: 1)
                var amt = amount; enc.setBytes(&amt, length: 4, index: 2)
                var vd = volumeDiff; enc.setBytes(&vd, length: 4, index: 3)
                var us = UInt32(stencil == nil ? 0 : 1); enc.setBytes(&us, length: 4, index: 4)
                PaletteUtil.dispatch(enc, compPSO, w, h)
                enc.endEncoding()
            }
            swap(&canvasA, &canvasB)
        }
        cb.commit()
        return NodeImage(texture: canvasA, indexTexture: idx, palette: palette, paletteCount: count)
    }
}

/// Pass 3: reintroduce pieces of the original through textured per-colour masks.
public struct CompOriginalOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let input = inputs["in"], let src = input.texture, let palette = input.palette,
              let original = inputs["original"]?.texture,
              let idx = try resolveIndex(input, ctx: ctx) else { return inputs["in"] }
        let w = src.width, h = src.height
        let count = input.paletteCount
        let stencil = inputs["stencil"]?.texture

        guard var canvasA = ctx.makeTexture(width: w, height: h),
              var canvasB = ctx.makeTexture(width: w, height: h),
              let mask = ctx.makeTexture(width: w, height: h),
              let blurred = ctx.makeTexture(width: w, height: h),
              let cb = ctx.queue.makeCommandBuffer() else { return inputs["in"] }

        try ctx.encode(kernel: "copy_tex", output: canvasA, inputs: [src], in: cb)
        let maskPSO = try ctx.pipeline("region_mask")
        let compPSO = try ctx.pipeline("comp_masked")
        let blur = MPSImageGaussianBlur(device: ctx.device, sigma: 6)
        blur.edgeMode = .clamp

        for i in 0..<count {
            if let enc = cb.makeComputeCommandEncoder() {
                enc.setComputePipelineState(maskPSO)
                enc.setTexture(mask, index: 0); enc.setTexture(idx, index: 1)
                var t = UInt32(i); enc.setBytes(&t, length: 4, index: 0)
                PaletteUtil.dispatch(enc, maskPSO, w, h)
                enc.endEncoding()
            }
            blur.encode(commandBuffer: cb, sourceTexture: mask, destinationTexture: blurred)
            if let enc = cb.makeComputeCommandEncoder() {
                enc.setComputePipelineState(compPSO)
                enc.setTexture(canvasB, index: 0); enc.setTexture(canvasA, index: 1)
                enc.setTexture(original, index: 2); enc.setTexture(blurred, index: 3)
                enc.setTexture(stencil ?? blurred, index: 4)
                var us = UInt32(stencil == nil ? 0 : 1); enc.setBytes(&us, length: 4, index: 0)
                PaletteUtil.dispatch(enc, compPSO, w, h)
                enc.endEncoding()
            }
            swap(&canvasA, &canvasB)
        }
        cb.commit()
        return NodeImage(texture: canvasA, indexTexture: idx, palette: palette, paletteCount: count)
    }
}

/// Glow: blend the image toward its blurred self.
public struct GlowOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let input = inputs["in"], let src = input.texture,
              let blurred = ctx.makeTexture(width: src.width, height: src.height),
              let out = ctx.makeTexture(width: src.width, height: src.height),
              let cb = ctx.queue.makeCommandBuffer() else { return inputs["in"] }
        let radius = max(0.5, Float(params["radius"]?.floatValue ?? 15))
        var amount = Float(params["amount"]?.floatValue ?? 0.3)
        let blur = MPSImageGaussianBlur(device: ctx.device, sigma: radius)
        blur.edgeMode = .clamp
        blur.encode(commandBuffer: cb, sourceTexture: src, destinationTexture: blurred)
        if let enc = cb.makeComputeCommandEncoder() {
            let pso = try ctx.pipeline("glow_add")
            enc.setComputePipelineState(pso)
            enc.setTexture(out, index: 0); enc.setTexture(src, index: 1); enc.setTexture(blurred, index: 2)
            enc.setBytes(&amount, length: 4, index: 0)
            PaletteUtil.dispatch(enc, pso, src.width, src.height)
            enc.endEncoding()
        }
        cb.commit()
        return NodeImage(texture: out, palette: input.palette, paletteCount: input.paletteCount)
    }
}
