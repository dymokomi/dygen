import XCTest
import Metal
@testable import DygenCore

/// Verifies compute/MPS ops run and produce sane output. Skipped without a GPU.
final class ComputeOpTests: XCTestCase {

    private func ctx() throws -> GPUContext {
        guard let c = GPUContext() else { throw XCTSkip("No Metal device") }
        return c
    }

    /// A small RGBA8 texture with a deterministic gradient for testing.
    private func gradientTexture(_ ctx: GPUContext, _ w: Int = 16, _ h: Int = 16) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .managed
        let tex = ctx.device.makeTexture(descriptor: desc)!
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                bytes[i] = UInt8(x * 255 / max(w - 1, 1))
                bytes[i+1] = UInt8(y * 255 / max(h - 1, 1))
                bytes[i+2] = 128
                bytes[i+3] = 255
            }
        }
        tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: bytes, bytesPerRow: w * 4)
        return tex
    }

    private func readback(_ tex: MTLTexture, ctx: GPUContext) -> [UInt8] {
        let w = tex.width, h = tex.height
        // Blit to a managed texture we can read.
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        desc.usage = [.shaderRead]
        desc.storageMode = .managed
        let dst = ctx.device.makeTexture(descriptor: desc)!
        let cb = ctx.queue.makeCommandBuffer()!
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: dst, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.synchronize(resource: dst)
        blit.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        dst.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        return bytes
    }

    func testPixelateBlocksAreUniform() throws {
        let c = try ctx()
        let src = gradientTexture(c, 16, 16)
        let out = try XCTUnwrap(PixelateOp().evaluate(inputs: ["in": NodeImage(texture: src)], params: ["size": .int(4)], ctx: c)?.texture)
        XCTAssertEqual(out.width, 16); XCTAssertEqual(out.height, 16)
        let px = readback(out, ctx: c)
        // Within a 4x4 block every pixel equals the block's top-left source pixel.
        func at(_ x: Int, _ y: Int) -> UInt8 { px[(y * 16 + x) * 4] } // red channel
        XCTAssertEqual(at(0, 0), at(3, 3))
        XCTAssertEqual(at(4, 0), at(7, 3))
        XCTAssertNotEqual(at(0, 0), at(4, 0)) // different blocks differ
    }

    func testBlurPreservesSizeAndZeroRadiusIsIdentity() throws {
        let c = try ctx()
        let src = gradientTexture(c, 16, 16)
        let blurred = try XCTUnwrap(BlurOp().evaluate(inputs: ["in": NodeImage(texture: src)], params: ["radius": .float(3)], ctx: c)?.texture)
        XCTAssertEqual(blurred.width, 16); XCTAssertEqual(blurred.height, 16)

        let identity = try XCTUnwrap(BlurOp().evaluate(inputs: ["in": NodeImage(texture: src)], params: ["radius": .float(0)], ctx: c)?.texture)
        let a = readback(src, ctx: c), b = readback(identity, ctx: c)
        XCTAssertEqual(a, b, "radius 0 should be a passthrough copy")
    }

    func testSharpenRuns() throws {
        let c = try ctx()
        let src = gradientTexture(c, 16, 16)
        let out = try XCTUnwrap(SharpenOp().evaluate(inputs: ["in": NodeImage(texture: src)], params: [:], ctx: c)?.texture)
        XCTAssertEqual(out.width, 16); XCTAssertEqual(out.height, 16)
    }
}
