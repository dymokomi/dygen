import Foundation
import CoreGraphics
import DygenCore

/// Builds the default painterly graph — the dygen.py pipeline as nodes, laid out
/// left→right, with Read/stencil pointing at the bundled sample assets so the
/// app opens to a working result. Returns the graph + the node to view (sharpen).
enum DefaultScene {
    private static func bundlePath(_ name: String) -> String {
        Bundle.main.url(forResource: name, withExtension: "png")?.path ?? ""
    }

    static func make() -> (graph: Graph, viewNode: UUID) {
        func node(_ type: String, _ x: CGFloat, _ y: CGFloat, _ params: [String: NodeValue] = [:]) -> Node {
            var n = NodeRegistry.descriptor(type)!.makeNode(at: CGPoint(x: x, y: y))
            for (k, v) in params { n.params[k] = v }
            return n
        }

        let read     = node("read", -560, -40, ["path": .string(bundlePath("ref"))])
        let stencil  = node("read", -560, 160, ["path": .string(bundlePath("tex1"))])
        let quantize = node("quantize", -380, -40, ["colors": .int(64)])
        let cleanup  = node("cleanup", -210, -40, ["radius": .int(13)])
        let pixelate = node("pixelate", -40, -40, ["size": .int(2)])
        let blobs    = node("paintBlobs", 130, -40)
        let brush    = node("brushStrokes", 300, 0)
        let comp     = node("compOriginal", 470, 0)
        let glow     = node("glow", 640, 0)
        let sharpen  = node("sharpen", 810, 0)
        let write    = node("write", 980, 0, ["path": .string("")])

        func e(_ a: Node, _ ap: String, _ b: Node, _ bp: String) -> Connection {
            Connection(fromNode: a.id, fromPort: ap, toNode: b.id, toPort: bp)
        }
        let graph = Graph(
            nodes: [read, stencil, quantize, cleanup, pixelate, blobs, brush, comp, glow, sharpen, write],
            connections: [
                e(read, "out", quantize, "in"),
                e(quantize, "out", cleanup, "in"),
                e(cleanup, "out", pixelate, "in"),
                e(pixelate, "out", blobs, "in"),
                e(blobs, "out", brush, "in"),
                e(stencil, "out", brush, "stencil"),
                e(brush, "out", comp, "in"),
                e(read, "out", comp, "original"),
                e(stencil, "out", comp, "stencil"),
                e(comp, "out", glow, "in"),
                e(glow, "out", sharpen, "in"),
                e(sharpen, "out", write, "in"),
            ]
        )
        return (graph, sharpen.id)
    }
}
