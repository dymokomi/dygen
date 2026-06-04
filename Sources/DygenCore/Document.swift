import Foundation
import CoreGraphics

/// Lightweight change notifier the executor (M4) subscribes to for dirty
/// tracking. The UI observes `Document` directly via `@Published`.
public final class ChangeBus {
    public enum Change: Equatable { case node(UUID); case graph }
    private var handlers: [(Change) -> Void] = []
    public init() {}
    public func subscribe(_ h: @escaping (Change) -> Void) { handlers.append(h) }
    public func didChange(_ id: UUID) { handlers.forEach { $0(.node(id)) } }
    public func didChangeGraph() { handlers.forEach { $0(.graph) } }
}

/// The single source of truth: holds the graph, selection, and the node whose
/// output the Canvas shows. All edits flow through `run(_:)` → a `Command`.
public final class Document: ObservableObject {
    @Published public var graph: Graph
    @Published public var selection: Set<UUID> = []
    @Published public var viewNodeID: UUID?
    @Published public var fileURL: URL?
    @Published public var isDirty: Bool = false

    public let commands = CommandBus()
    public let changes = ChangeBus()

    public init(graph: Graph = Graph()) {
        self.graph = graph
    }

    public func markDirty() { isDirty = true }

    // Editing entry points.
    public func run(_ cmd: Command) { commands.run(cmd, on: self) }
    public func undo() { commands.undo(on: self) }
    public func redo() { commands.redo(on: self) }
}

/// The persisted `.dygen` document body.
struct DocumentFile: Codable {
    var version: Int = 1
    var viewNode: UUID?
    var graph: Graph
}

public extension Document {
    func save(to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try enc.encode(DocumentFile(viewNode: viewNodeID, graph: graph))
        try data.write(to: url)
        fileURL = url
        isDirty = false
    }

    static func load(from url: URL) throws -> Document {
        let file = try JSONDecoder().decode(DocumentFile.self, from: Data(contentsOf: url))
        let doc = Document(graph: file.graph)
        doc.viewNodeID = file.viewNode
        doc.fileURL = url
        return doc
    }

    /// Replace this document's contents in place (used by Open/New so the shared
    /// `Document` object reference stays stable for the UI).
    func replaceContents(graph: Graph, viewNode: UUID?, fileURL: URL?) {
        self.graph = graph
        self.viewNodeID = viewNode
        self.fileURL = fileURL
        self.selection = []
        self.isDirty = false
        commands.clear()
    }

    func reset() {
        replaceContents(graph: Graph(), viewNode: nil, fileURL: nil)
    }
}
