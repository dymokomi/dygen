import XCTest
import Metal
@testable import DygenCore

final class QuantizeTests: XCTestCase {

    private func ctx() throws -> GPUContext {
        guard let c = GPUContext() else { throw XCTSkip("No Metal device") }
        return c
    }

    private var ref: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("ref/ref.png")
    }

    func testKMeansFindsKCentroids() {
        // Two well-separated clusters → 2 centroids near them.
        var samples: [SIMD3<Float>] = []
        for _ in 0..<100 { samples.append(SIMD3(0.1, 0.1, 0.1)) }
        for _ in 0..<100 { samples.append(SIMD3(0.9, 0.9, 0.9)) }
        let c = KMeans.centroids(samples: samples, k: 2)
        XCTAssertEqual(c.count, 2)
        let lo = c.min { $0.x < $1.x }!, hi = c.max { $0.x < $1.x }!
        XCTAssertLessThan(lo.x, 0.3)
        XCTAssertGreaterThan(hi.x, 0.7)
    }

    func testQuantizeReducesColorCount() throws {
        let c = try ctx()
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ref.path), "ref.png missing")
        let img = try XCTUnwrap(ReadOp().evaluate(inputs: [:], params: ["path": .string(ref.path)], ctx: c))
        let q = try XCTUnwrap(QuantizeOp().evaluate(inputs: ["in": img], params: ["colors": .int(8)], ctx: c))
        XCTAssertNotNil(q.indexTexture)
        XCTAssertNotNil(q.palette)
        XCTAssertLessThanOrEqual(q.paletteCount, 8)

        // The output image should contain at most `paletteCount` distinct colours.
        let (bytes, w, h) = TextureIO.readbackRGBA8(q.texture!, ctx: c)
        var distinct = Set<UInt32>()
        var p = 0
        while p < w * h {
            let i = p * 4
            distinct.insert(UInt32(bytes[i]) << 16 | UInt32(bytes[i+1]) << 8 | UInt32(bytes[i+2]))
            p += 97 // sparse sample
        }
        XCTAssertLessThanOrEqual(distinct.count, 8)
    }

    func testCleanupRunsAndPreservesPalette() throws {
        let c = try ctx()
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ref.path), "ref.png missing")
        let img = try XCTUnwrap(ReadOp().evaluate(inputs: [:], params: ["path": .string(ref.path)], ctx: c))
        let q = try XCTUnwrap(QuantizeOp().evaluate(inputs: ["in": img], params: ["colors": .int(16)], ctx: c))
        let cleaned = try XCTUnwrap(CleanupOp().evaluate(inputs: ["in": q], params: ["radius": .int(5)], ctx: c))
        XCTAssertEqual(cleaned.texture?.width, q.texture?.width)
        XCTAssertEqual(cleaned.paletteCount, q.paletteCount)
        XCTAssertNotNil(cleaned.indexTexture)
    }
}
