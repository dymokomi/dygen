import Metal
import MetalPerformanceShaders
import Foundation

/// Block pixelation (custom compute kernel).
public struct PixelateOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let input = inputs["in"], let src = input.texture,
              let out = ctx.makeTexture(width: src.width, height: src.height) else { return nil }
        var block = UInt32(max(1, params["size"]?.intValue ?? 2))
        try ctx.runKernel("pixelate", output: out, inputs: [src]) { enc in
            enc.setBytes(&block, length: MemoryLayout<UInt32>.size, index: 0)
        }
        // Pixelate the index too so masks stay aligned with the blobs.
        var outIdx: MTLTexture?
        if let idx = input.indexTexture,
           let oi = ctx.makeTexture(width: src.width, height: src.height, pixelFormat: .r8Uint) {
            try ctx.runKernel("pixelate_index", output: oi, inputs: [idx]) { enc in
                enc.setBytes(&block, length: MemoryLayout<UInt32>.size, index: 0)
            }
            outIdx = oi
        }
        return NodeImage(texture: out, indexTexture: outIdx, palette: input.palette, paletteCount: input.paletteCount)
    }
}

/// Gaussian blur (MPS).
public struct BlurOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let src = inputs["in"]?.texture, let out = ctx.makeTexture(width: src.width, height: src.height),
              let cb = ctx.queue.makeCommandBuffer() else { return nil }
        let radius = Float(params["radius"]?.floatValue ?? 4)
        if radius <= 0.01 {
            try ctx.encode(kernel: "copy_tex", output: out, inputs: [src], in: cb)
        } else {
            let blur = MPSImageGaussianBlur(device: ctx.device, sigma: radius)
            blur.edgeMode = .clamp
            blur.encode(commandBuffer: cb, sourceTexture: src, destinationTexture: out)
        }
        cb.commit()
        return NodeImage(texture: out)
    }
}

/// Sharpen — PIL's SHARPEN 3×3 kernel (MPS convolution).
public struct SharpenOp: NodeOp {
    public init() {}
    private static let weights: [Float] = [-2, -2, -2, -2, 32, -2, -2, -2, -2].map { $0 / 16.0 }
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let src = inputs["in"]?.texture, let out = ctx.makeTexture(width: src.width, height: src.height),
              let cb = ctx.queue.makeCommandBuffer() else { return nil }
        let conv = MPSImageConvolution(device: ctx.device, kernelWidth: 3, kernelHeight: 3, weights: Self.weights)
        conv.edgeMode = .clamp
        conv.encode(commandBuffer: cb, sourceTexture: src, destinationTexture: out)
        cb.commit()
        return NodeImage(texture: out)
    }
}
