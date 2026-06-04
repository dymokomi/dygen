import XCTest
import CoreGraphics
@testable import DygenCore

final class DygenCoreTests: XCTestCase {

    override func setUp() {
        super.setUp()
        NodeRegistry.registerBuiltins()
    }

    // MARK: - Smoke

    func testBootstrap() {
        XCTAssertEqual(DygenCore.name, "DygenCore")
        DygenCore.bootstrap()
        XCTAssertNotNil(NodeRegistry.descriptor("read"))
        XCTAssertNotNil(NodeRegistry.descriptor("write"))
    }

    // MARK: - NodeValue codable

    func testNodeValueRoundTrip() throws {
        let values: [NodeValue] = [.float(1.5), .int(64), .bool(true),
                                   .color(0.1, 0.2, 0.3), .float3(1, 2, 3), .string("ref/ref.png")]
        for v in values {
            let data = try JSONEncoder().encode(v)
            let back = try JSONDecoder().decode(NodeValue.self, from: data)
            XCTAssertEqual(v, back)
        }
    }

    // MARK: - Document save / load

    func testDocumentSaveLoadRoundTrip() throws {
        let read = NodeRegistry.descriptor("read")!.makeNode(at: CGPoint(x: 10, y: 20))
        var write = NodeRegistry.descriptor("write")!.makeNode(at: CGPoint(x: 200, y: 20))
        write.params["path"] = .string("out.png")
        let conn = Connection(fromNode: read.id, fromPort: "out", toNode: write.id, toPort: "in")
        let doc = Document(graph: Graph(nodes: [read, write], connections: [conn]))
        doc.viewNodeID = write.id

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).dygen")
        defer { try? FileManager.default.removeItem(at: url) }
        try doc.save(to: url)

        let loaded = try Document.load(from: url)
        XCTAssertEqual(loaded.graph, doc.graph)
        XCTAssertEqual(loaded.viewNodeID, write.id)
        XCTAssertFalse(loaded.isDirty)
    }

    // MARK: - Commands & undo

    func testAddDeleteUndoRedo() {
        let doc = Document()
        let node = NodeRegistry.descriptor("read")!.makeNode()
        doc.run(AddNode(node: node))
        XCTAssertEqual(doc.graph.nodes.count, 1)
        XCTAssertTrue(doc.isDirty)

        doc.undo()
        XCTAssertEqual(doc.graph.nodes.count, 0)
        doc.redo()
        XCTAssertEqual(doc.graph.nodes.count, 1)
    }

    func testDeleteNodeRemovesAndRestoresConnections() {
        let a = NodeRegistry.descriptor("read")!.makeNode()
        let b = NodeRegistry.descriptor("write")!.makeNode()
        let conn = Connection(fromNode: a.id, fromPort: "out", toNode: b.id, toPort: "in")
        let doc = Document(graph: Graph(nodes: [a, b], connections: [conn]))

        doc.run(DeleteNode(nodeID: a.id))
        XCTAssertEqual(doc.graph.nodes.count, 1)
        XCTAssertEqual(doc.graph.connections.count, 0) // edge removed with the node

        doc.undo()
        XCTAssertEqual(doc.graph.nodes.count, 2)
        XCTAssertEqual(doc.graph.connections.count, 1) // edge restored
    }

    func testSetParamUndoRestoresOldValue() {
        var node = NodeRegistry.descriptor("read")!.makeNode()
        node.params["path"] = .string("a.png")
        let doc = Document(graph: Graph(nodes: [node]))

        doc.run(SetParam(nodeID: node.id, key: "path", value: .string("b.png")))
        XCTAssertEqual(doc.graph.node(node.id)?.params["path"], .string("b.png"))
        doc.undo()
        XCTAssertEqual(doc.graph.node(node.id)?.params["path"], .string("a.png"))
    }

    func testSetParamMergesContinuousEdits() {
        let node = NodeRegistry.descriptor("read")!.makeNode()
        let doc = Document(graph: Graph(nodes: [node]))
        // Simulate a drag: many merged SetParams = one undo step back to original.
        for v in stride(from: Float(0), through: 1, by: 0.1) {
            doc.run(SetParam(nodeID: node.id, key: "x", value: .float(v), mergeKey: "drag-x"))
        }
        XCTAssertEqual(doc.commands.undoStack.count, 1)
        doc.undo()
        XCTAssertNil(doc.graph.node(node.id)?.params["x"]) // back to no value
    }

    // MARK: - Connect & cycles

    func testConnectDisplacesExistingInput() {
        let a = NodeRegistry.descriptor("read")!.makeNode()
        let a2 = NodeRegistry.descriptor("read")!.makeNode()
        let b = NodeRegistry.descriptor("write")!.makeNode()
        let doc = Document(graph: Graph(nodes: [a, a2, b]))

        doc.run(Connect(connection: Connection(fromNode: a.id, fromPort: "out", toNode: b.id, toPort: "in")))
        doc.run(Connect(connection: Connection(fromNode: a2.id, fromPort: "out", toNode: b.id, toPort: "in")))
        XCTAssertEqual(doc.graph.connections.count, 1) // second displaced the first
        XCTAssertEqual(doc.graph.inputConnection(to: b.id, port: "in")?.fromNode, a2.id)

        doc.undo() // restores the first connection
        XCTAssertEqual(doc.graph.inputConnection(to: b.id, port: "in")?.fromNode, a.id)
    }

    // MARK: - Viewport

    func testZoomKeepsWorldPointUnderCursor() {
        var vp = NodeGraphViewport(zoom: 1, panX: 0, panY: 0, width: 800, height: 600)
        let cursor = CGPoint(x: 520, y: 240)
        let before = vp.screenToWorld(cursor.x, cursor.y)
        vp.applyZoom(2.3, around: cursor)
        let after = vp.screenToWorld(cursor.x, cursor.y)
        XCTAssertEqual(before.x, after.x, accuracy: 0.001)
        XCTAssertEqual(before.y, after.y, accuracy: 0.001)
        XCTAssertEqual(vp.zoom, 2.3, accuracy: 0.001)
    }

    func testScreenWorldRoundTrip() {
        let vp = NodeGraphViewport(zoom: 1.7, panX: 33, panY: -12, width: 640, height: 480)
        let w = vp.screenToWorld(210, 305)
        let s = vp.worldToScreen(w.x, w.y)
        XCTAssertEqual(s.x, 210, accuracy: 0.001)
        XCTAssertEqual(s.y, 305, accuracy: 0.001)
    }

    func testCycleDetection() {
        let a = NodeRegistry.descriptor("write")!.makeNode()
        let b = NodeRegistry.descriptor("write")!.makeNode()
        var g = Graph(nodes: [a, b])
        g.connections.append(Connection(fromNode: a.id, fromPort: "out", toNode: b.id, toPort: "in"))
        XCTAssertTrue(g.wouldCreateCycle(source: b.id, target: a.id))  // b->a closes a loop
        XCTAssertFalse(g.wouldCreateCycle(source: a.id, target: b.id)) // already exists, not a cycle
    }
}
