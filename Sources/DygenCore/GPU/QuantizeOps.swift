import Metal
import simd
import Foundation

/// Capped at 64 so the cleanup mode-filter's per-thread count array fits.
let kMaxPalette = 64

/// Lightweight k-means over sampled pixels (centroids in 0…1 RGB).
enum KMeans {
    static func centroids(samples: [SIMD3<Float>], k: Int, iterations: Int = 12) -> [SIMD3<Float>] {
        guard !samples.isEmpty else { return [] }
        let kk = max(1, min(k, samples.count))
        var c: [SIMD3<Float>] = (0..<kk).map { samples[$0 * samples.count / kk] }
        for _ in 0..<iterations {
            var sums = [SIMD3<Float>](repeating: .zero, count: kk)
            var counts = [Int](repeating: 0, count: kk)
            for s in samples {
                var best = 0; var bd = Float.greatestFiniteMagnitude
                for i in 0..<kk { let d = s - c[i]; let dist = simd_dot(d, d); if dist < bd { bd = dist; best = i } }
                sums[best] += s; counts[best] += 1
            }
            for i in 0..<kk where counts[i] > 0 { c[i] = sums[i] / Float(counts[i]) }
        }
        return c
    }
}

/// Helpers shared by quantize-family ops.
enum PaletteUtil {
    /// Build a Metal buffer of float4 (rgb,1) palette entries.
    static func buffer(_ palette: [SIMD3<Float>], ctx: GPUContext) -> MTLBuffer? {
        let entries = palette.map { SIMD4<Float>($0.x, $0.y, $0.z, 1) }
        return ctx.device.makeBuffer(bytes: entries, length: max(1, entries.count) * MemoryLayout<SIMD4<Float>>.stride,
                                     options: .storageModeShared)
    }

    /// Recompute an index texture from an RGB texture + palette.
    static func indexTexture(rgb: MTLTexture, palette: MTLBuffer, count: Int, ctx: GPUContext) throws -> MTLTexture {
        guard let idx = ctx.makeTexture(width: rgb.width, height: rgb.height, pixelFormat: .r8Uint),
              let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw NodeOpError.io("indexTexture: alloc failed")
        }
        let pso = try ctx.pipeline("assign_index")
        enc.setComputePipelineState(pso)
        enc.setTexture(idx, index: 0)
        enc.setTexture(rgb, index: 1)
        enc.setBuffer(palette, offset: 0, index: 0)
        var c = UInt32(count); enc.setBytes(&c, length: 4, index: 1)
        dispatch(enc, pso, rgb.width, rgb.height)
        enc.endEncoding(); cb.commit()
        return idx
    }

    static func dispatch(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, _ w: Int, _ h: Int) {
        let tw = pso.threadExecutionWidth
        let th = max(1, pso.maxTotalThreadsPerThreadgroup / tw)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    }
}

/// Reduce to N colours via k-means; outputs the quantized RGB, an index texture,
/// and the palette buffer.
public struct QuantizeOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let src = inputs["in"]?.texture else { return nil }
        let k = min(kMaxPalette, max(2, params["colors"]?.intValue ?? 64))

        // Sample pixels on the CPU and k-means for the palette.
        let (bytes, w, h) = TextureIO.readbackRGBA8(src, ctx: ctx)
        guard !bytes.isEmpty else { return nil }
        let stride = max(1, (w * h) / 4096)
        var samples: [SIMD3<Float>] = []
        samples.reserveCapacity(4096)
        var p = 0
        while p < w * h {
            let i = p * 4
            samples.append(SIMD3(Float(bytes[i]) / 255, Float(bytes[i+1]) / 255, Float(bytes[i+2]) / 255))
            p += stride
        }
        let palette = KMeans.centroids(samples: samples, k: k)
        guard let palBuf = PaletteUtil.buffer(palette, ctx: ctx),
              let outColor = ctx.makeTexture(width: w, height: h),
              let outIdx = ctx.makeTexture(width: w, height: h, pixelFormat: .r8Uint),
              let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }

        let pso = try ctx.pipeline("quantize_assign")
        enc.setComputePipelineState(pso)
        enc.setTexture(outColor, index: 0)
        enc.setTexture(outIdx, index: 1)
        enc.setTexture(src, index: 2)
        enc.setBuffer(palBuf, offset: 0, index: 0)
        var count = UInt32(palette.count); enc.setBytes(&count, length: 4, index: 1)
        PaletteUtil.dispatch(enc, pso, w, h)
        enc.endEncoding(); cb.commit()

        return NodeImage(texture: outColor, indexTexture: outIdx, palette: palBuf, paletteCount: palette.count)
    }
}

/// Mode filter over the palette indices (despeckle). Passes the palette through.
public struct CleanupOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let input = inputs["in"], let src = input.texture else { return nil }
        // Need a palette to operate on indices; otherwise pass through unchanged.
        guard let palette = input.palette, input.paletteCount > 0 else { return input }
        let radius = max(0, min(40, params["radius"]?.intValue ?? 13))
        if radius == 0 { return input }

        let idx = try input.indexTexture ?? PaletteUtil.indexTexture(rgb: src, palette: palette, count: input.paletteCount, ctx: ctx)
        guard let outColor = ctx.makeTexture(width: src.width, height: src.height),
              let outIdx = ctx.makeTexture(width: src.width, height: src.height, pixelFormat: .r8Uint),
              let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }

        let pso = try ctx.pipeline("cleanup_mode")
        enc.setComputePipelineState(pso)
        enc.setTexture(outColor, index: 0)
        enc.setTexture(outIdx, index: 1)
        enc.setTexture(idx, index: 2)
        enc.setBuffer(palette, offset: 0, index: 0)
        var count = UInt32(input.paletteCount); enc.setBytes(&count, length: 4, index: 1)
        var r = Int32(radius); enc.setBytes(&r, length: 4, index: 2)
        PaletteUtil.dispatch(enc, pso, src.width, src.height)
        enc.endEncoding(); cb.commit()

        return NodeImage(texture: outColor, indexTexture: outIdx, palette: palette, paletteCount: input.paletteCount)
    }
}
