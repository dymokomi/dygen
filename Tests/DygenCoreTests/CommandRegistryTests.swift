import XCTest
import Metal
@testable import DygenCore

final class CommandRegistryTests: XCTestCase {

    override func setUp() {
        super.setUp()
        NodeRegistry.registerBuiltins()
        CommandRegistry.shared.registerBuiltins()
    }

    func testDispatchSetParamByNodeType() throws {
        let quantize = NodeRegistry.descriptor("quantize")!.makeNode()
        let doc = Document(graph: Graph(nodes: [quantize]))

        let params = SetParamCommand(node: "quantize", key: "colors", value: .int(16))
        let json = try JSONEncoder().encode(params)
        try CommandRegistry.shared.dispatch(name: "setParam", paramsJSON: json, on: doc)

        XCTAssertEqual(doc.graph.node(quantize.id)?.params["colors"], .int(16))
        doc.undo()
        XCTAssertEqual(doc.graph.node(quantize.id)?.params["colors"], .int(64)) // default restored
    }

    func testDispatchUnknownCommandThrows() {
        let doc = Document()
        XCTAssertThrowsError(try CommandRegistry.shared.dispatch(name: "nope", paramsJSON: Data("{}".utf8), on: doc))
    }

    func testAddNodeCommand() throws {
        let doc = Document()
        let json = try JSONEncoder().encode(AddNodeCommand(type: "blur", x: 10, y: 20))
        try CommandRegistry.shared.dispatch(name: "addNode", paramsJSON: json, on: doc)
        XCTAssertEqual(doc.graph.nodes.count, 1)
        XCTAssertEqual(doc.graph.nodes.first?.type, "blur")
    }

    // MARK: - Headless

    func testHeadlessSetAndExport() throws {
        guard GPUContext() != nil else { throw XCTSkip("No Metal device") }
        let ref = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("ref/ref.png")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ref.path), "ref.png missing")

        // A minimal default graph: read → quantize, viewing quantize.
        var read = NodeRegistry.descriptor("read")!.makeNode()
        read.params["path"] = .string(ref.path)
        let quantize = NodeRegistry.descriptor("quantize")!.makeNode()
        let graph = Graph(nodes: [read, quantize],
                          connections: [Connection(fromNode: read.id, fromPort: "out", toNode: quantize.id, toPort: "in")])

        let out = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).png")
        defer { try? FileManager.default.removeItem(at: out) }

        let code = Headless.run(["--set", "quantize", "colors", "8", "--view", "quantize", "--export", out.path]) {
            (graph, quantize.id)
        }
        XCTAssertEqual(code, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: out.path), "headless --export should write a PNG")
    }
}
