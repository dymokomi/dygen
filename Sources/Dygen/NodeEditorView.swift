import SwiftUI
import DygenCore

/// SwiftUI host for the AppKit node canvas. Pushes the document's graph/selection
/// into the canvas and routes the canvas's commands back through the document.
struct NodeEditorView: NSViewRepresentable {
    @ObservedObject var document: Document

    func makeNSView(context: Context) -> NodeCanvasView {
        let view = NodeCanvasView()
        view.onRun = { [weak document] cmd in document?.run(cmd) }
        view.onSelect = { [weak document] sel in document?.selection = sel }
        view.onSetViewNode = { [weak document] id in document?.viewNodeID = id }
        view.apply(nodes: document.graph.nodes, connections: document.graph.connections,
                   selected: document.selection, viewNode: document.viewNodeID)
        return view
    }

    func updateNSView(_ view: NodeCanvasView, context: Context) {
        view.apply(nodes: document.graph.nodes, connections: document.graph.connections,
                   selected: document.selection, viewNode: document.viewNodeID)
    }
}
