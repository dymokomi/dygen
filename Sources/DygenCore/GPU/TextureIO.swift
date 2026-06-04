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
}
