import XCTest
import Metal
@testable import DygenCore

/// Exercises the real GPU path: load ref/ref.png into a Metal texture (Read),
/// write it back to PNG (Write), and drive both through the Executor with
/// caching. Skipped when no Metal device is present.
final class GPUTests: XCTestCase {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // DygenCoreTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
    }

    func testReadWritePNGRoundTrip() throws {
        guard let gpu = GPUContext() else { throw XCTSkip("No Metal device") }
        let ref = repoRoot.appendingPathComponent("ref/ref.png")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ref.path), "ref/ref.png missing")

        let tex = try XCTUnwrap(ReadOp().evaluate(inputs: [:], params: ["path": .string(ref.path)], ctx: gpu))
        XCTAssertEqual(tex.width, 1024)
        XCTAssertEqual(tex.height, 1024)

        let out = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).png")
        defer { try? FileManager.default.removeItem(at: out) }
        _ = try WriteOp().evaluate(inputs: ["in": tex], params: ["path": .string(out.path)], ctx: gpu)
        XCTAssertTrue(FileManager.default.fileExists(atPath: out.path))

        // The written PNG re-reads at the same dimensions.
        let back = try XCTUnwrap(ReadOp().evaluate(inputs: [:], params: ["path": .string(out.path)], ctx: gpu))
        XCTAssertEqual(back.width, 1024)
        XCTAssertEqual(back.height, 1024)
    }

    func testExecutorCachesReadOutput() throws {
        guard let gpu = GPUContext() else { throw XCTSkip("No Metal device") }
        NodeRegistry.registerBuiltins()
        let ref = repoRoot.appendingPathComponent("ref/ref.png")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ref.path), "ref/ref.png missing")

        var read = NodeRegistry.descriptor("read")!.makeNode()
        read.params["path"] = .string(ref.path)
        let graph = Graph(nodes: [read])
        let ex = Executor(ctx: gpu, opForType: BuiltinOps.op(for:))

        let a = try ex.evaluate(read.id, graph: graph)
        let b = try ex.evaluate(read.id, graph: graph)
        XCTAssertNotNil(a)
        XCTAssertTrue(a === b, "second evaluation should return the cached texture")

        // After marking dirty, it recomputes (new instance).
        ex.markDirty(read.id, graph: graph)
        let c = try ex.evaluate(read.id, graph: graph)
        XCTAssertFalse(a === c, "dirty node should re-evaluate")
    }
}
