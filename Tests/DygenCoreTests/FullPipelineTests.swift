import XCTest
import Metal
@testable import DygenCore

/// Builds the full painterly graph (the dygen.py pipeline as nodes) and renders
/// ref/ref.png to a PNG, verifying it runs end-to-end and is fast.
final class FullPipelineTests: XCTestCase {

    private var root: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    func testFullPainterlyPipeline() throws {
        guard let gpu = GPUContext() else { throw XCTSkip("No Metal device") }
        NodeRegistry.registerBuiltins()
        let ref = root.appendingPathComponent("ref/ref.png")
        let stencil = root.appendingPathComponent("tex/tex1.png")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ref.path), "ref/ref.png missing")

        func node(_ type: String, _ params: [String: NodeValue] = [:]) -> Node {
            var n = NodeRegistry.descriptor(type)!.makeNode()
            for (k, v) in params { n.params[k] = v }
            return n
        }

        let read     = node("read", ["path": .string(ref.path)])
        let stencilN  = node("read", ["path": .string(stencil.path)])
        let quantize = node("quantize", ["colors": .int(64)])
        let cleanup  = node("cleanup", ["radius": .int(13)])
        let pixelate = node("pixelate", ["size": .int(2)])
        let blobs    = node("paintBlobs", ["seed": .int(2)])
        let brush    = node("brushStrokes")
        let comp     = node("compOriginal")
        let glow     = node("glow")
        let sharpen  = node("sharpen")
        let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("dygen_full.png")
        let write    = node("write", ["path": .string(outURL.path)])

        func edge(_ a: Node, _ ap: String, _ b: Node, _ bp: String) -> Connection {
            Connection(fromNode: a.id, fromPort: ap, toNode: b.id, toPort: bp)
        }
        let graph = Graph(
            nodes: [read, stencilN, quantize, cleanup, pixelate, blobs, brush, comp, glow, sharpen, write],
            connections: [
                edge(read, "out", quantize, "in"),
                edge(quantize, "out", cleanup, "in"),
                edge(cleanup, "out", pixelate, "in"),
                edge(pixelate, "out", blobs, "in"),
                edge(blobs, "out", brush, "in"),
                edge(stencilN, "out", brush, "stencil"),
                edge(brush, "out", comp, "in"),
                edge(read, "out", comp, "original"),
                edge(stencilN, "out", comp, "stencil"),
                edge(comp, "out", glow, "in"),
                edge(glow, "out", sharpen, "in"),
                edge(sharpen, "out", write, "in"),
            ]
        )

        let ex = Executor(ctx: gpu, opForType: BuiltinOps.op(for:))

        // Render and time it.
        let start = Date()
        try ex.evaluate(write.id, graph: graph)
        let elapsed = Date().timeIntervalSince(start)
        print("Full pipeline render: \(String(format: "%.3f", elapsed))s → \(outURL.path)")

        XCTAssertTrue(FileManager.default.fileExists(atPath: outURL.path), "output PNG should exist")
        let back = try XCTUnwrap(ReadOp().evaluate(inputs: [:], params: ["path": .string(outURL.path)], ctx: gpu)?.texture)
        XCTAssertEqual(back.width, 1024)
        XCTAssertEqual(back.height, 1024)
        XCTAssertLessThan(elapsed, 5.0, "full pipeline should be well under 5s (vs Python's minutes)")
    }
}
