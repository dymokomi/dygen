import Metal
import MetalKit
import Foundation

/// Load an image file into a texture usable by the display + later compute ops.
public struct ReadOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let path = params["path"]?.stringValue, !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NodeOpError.io("Read: file not found at \(url.path)")
        }
        let loader = MTKTextureLoader(device: ctx.device)
        let usage = MTLTextureUsage([.shaderRead, .shaderWrite, .renderTarget])
        let tex = try loader.newTexture(URL: url, options: [
            .SRGB: NSNumber(value: false),
            .textureUsage: NSNumber(value: usage.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue),
        ])
        return NodeImage(texture: tex)
    }
}

/// Write the input texture to a PNG file (a sink — no output).
public struct WriteOp: NodeOp {
    public init() {}
    public func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage? {
        guard let tex = inputs["in"]?.texture else { return nil }
        guard let path = params["path"]?.stringValue, !path.isEmpty else {
            throw NodeOpError.missingParam("Write: no output path set")
        }
        try TextureIO.writePNG(tex, to: URL(fileURLWithPath: path), ctx: ctx)
        return nil
    }
}

public enum BuiltinOps {
    public static func op(for type: String) -> NodeOp? {
        switch type {
        case "read":     return ReadOp()
        case "write":    return WriteOp()
        case "pixelate": return PixelateOp()
        case "blur":     return BlurOp()
        case "sharpen":  return SharpenOp()
        case "quantize":     return QuantizeOp()
        case "cleanup":      return CleanupOp()
        case "paintBlobs":   return PaintBlobsOp()
        case "brushStrokes": return BrushStrokesOp()
        case "compOriginal": return CompOriginalOp()
        case "glow":         return GlowOp()
        default:             return nil
        }
    }
}
