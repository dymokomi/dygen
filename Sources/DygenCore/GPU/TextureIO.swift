import Metal
import CoreImage
import Foundation

/// Read back a Metal texture and encode it as a PNG. Uses Core Image only for
/// GPU→file readback (display + processing stay pure Metal).
public enum TextureIO {
    public static func writePNG(_ tex: MTLTexture, to url: URL, ctx: GPUContext) throws {
        guard let ci = CIImage(mtlTexture: tex, options: [.colorSpace: CGColorSpaceCreateDeviceRGB()]) else {
            throw NodeOpError.io("Write: could not read texture")
        }
        // Metal texture origin is top-left; CIImage is bottom-left — flip.
        let flipped = ci.transformed(by: CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -ci.extent.height))
        let cictx = CIContext(mtlDevice: ctx.device)
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        try cictx.writePNGRepresentation(of: flipped, to: url, format: .RGBA8, colorSpace: cs)
    }

    /// Copy a texture into CPU-readable RGBA8 bytes (blit → managed → getBytes).
    public static func readbackRGBA8(_ tex: MTLTexture, ctx: GPUContext) -> (bytes: [UInt8], width: Int, height: Int) {
        let w = tex.width, h = tex.height
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        desc.usage = [.shaderRead]
        desc.storageMode = .managed
        guard let dst = ctx.device.makeTexture(descriptor: desc),
              let cb = ctx.queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return ([], w, h) }
        blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: dst, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.synchronize(resource: dst)
        blit.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        dst.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        return (bytes, w, h)
    }
}
