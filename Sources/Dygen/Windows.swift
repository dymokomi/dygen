import SwiftUI
import WindowKit
import DygenCore

/// Dygen's concrete window kinds and the default docking layout.
enum DygenWindows {
    static let canvas     = WindowKind("canvas")
    static let nodeEditor = WindowKind("nodeEditor")
    static let properties = WindowKind("properties")
    static let log        = WindowKind("log")

    static let registry = WindowKindRegistry([
        WindowKindInfo(canvas,     title: "Canvas",      systemImage: "photo"),
        WindowKindInfo(nodeEditor, title: "Node Editor", systemImage: "point.3.connected.trianglepath.dotted"),
        WindowKindInfo(properties, title: "Properties",  systemImage: "slider.horizontal.3"),
        WindowKindInfo(log,        title: "Log",         systemImage: "text.alignleft"),
    ])

    /// Default arrangement: node editor + log on the left, canvas center,
    /// properties right.
    static func defaultWorkspace() -> Workspace {
        let leftColumn = LayoutNode.split(SplitState(
            axis: .vertical,
            children: [.area(AreaState(kind: nodeEditor)), .area(AreaState(kind: log))],
            fractions: [0.7, 0.3]
        ))
        let root = LayoutNode.split(SplitState(
            axis: .horizontal,
            children: [leftColumn, .area(AreaState(kind: canvas)), .area(AreaState(kind: properties))],
            fractions: [0.5, 0.3, 0.2]
        ))
        return Workspace(name: "Default", rootNode: root)
    }

    /// ~/Library/Application Support/Dygen/layout.json
    static func storeURL() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
        return base.appendingPathComponent("Dygen/layout.json")
    }

    static func makeLayoutManager() -> LayoutManager {
        LayoutManager(registry: registry, storeURL: storeURL(), defaultWorkspace: defaultWorkspace())
    }

    /// A fresh document seeded with a Read → (gap) → Write so the editor isn't
    /// empty on first launch.
    static func makeDocument() -> Document {
        NodeRegistry.registerBuiltins()
        let read = NodeRegistry.descriptor("read")!.makeNode(at: CGPoint(x: -230, y: -30))
        let write = NodeRegistry.descriptor("write")!.makeNode(at: CGPoint(x: 120, y: -30))
        return Document(graph: Graph(nodes: [read, write]))
    }

    /// Editor view for a window kind.
    static func editor(for kind: WindowKind, model: AppModel) -> AnyView {
        let document = model.document
        if kind == nodeEditor { return AnyView(NodeEditorView(document: document)) }
        if kind == properties { return AnyView(PropertiesView(document: document)) }
        if kind == log        { return AnyView(LogView()) }
        if kind == canvas {
            if let gpu = model.gpu, let ex = model.executor {
                return AnyView(CanvasView(document: document, gpu: gpu, executor: ex))
            }
            return AnyView(AreaEditorPlaceholder(title: "Canvas (no GPU)", systemImage: "exclamationmark.triangle"))
        }
        let info = registry.info(for: kind)
        return AnyView(AreaEditorPlaceholder(title: info?.title ?? kind.id,
                                             systemImage: info?.systemImage ?? "square.dashed"))
    }
}
