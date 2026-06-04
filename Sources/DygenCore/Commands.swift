import Foundation
import CoreGraphics

/// A reversible edit. Following dray's pattern, `perform` mutates the document
/// and returns *its own inverse* — no separate snapshot store.
public protocol Command {
    var label: String { get }
    /// Continuous edits (e.g. a slider drag) sharing a merge key coalesce into
    /// one undo step. Nil = always a discrete step.
    var mergeKey: String? { get }
    @discardableResult func perform(_ doc: Document) -> Command
}

public extension Command {
    var mergeKey: String? { nil }
}

/// Undo/redo engine. Each entry on the stacks is the inverse to apply.
public final class CommandBus {
    public private(set) var undoStack: [Command] = []
    public private(set) var redoStack: [Command] = []
    private let limit = 100

    public init() {}

    public var canUndo: Bool { !undoStack.isEmpty }
    public var canRedo: Bool { !redoStack.isEmpty }

    public func run(_ cmd: Command, on doc: Document) {
        let inverse = cmd.perform(doc)
        AppLog.shared.command(cmd.label)
        if let key = cmd.mergeKey, let last = undoStack.last, last.mergeKey == key {
            // Coalesce: keep the earlier (pre-drag) inverse, drop this one.
        } else {
            undoStack.append(inverse)
            if undoStack.count > limit { undoStack.removeFirst(undoStack.count - limit) }
        }
        redoStack.removeAll()
    }

    public func undo(on doc: Document) {
        guard let inverse = undoStack.popLast() else { return }
        redoStack.append(inverse.perform(doc))
    }

    public func redo(on doc: Document) {
        guard let redo = redoStack.popLast() else { return }
        undoStack.append(redo.perform(doc))
    }

    public func clear() {
        undoStack.removeAll()
        redoStack.removeAll()
    }

    /// Register an undo step for a change that was already applied live (e.g. a
    /// slider drag that updated the model directly). `inverse` should restore
    /// the pre-edit state; undo/redo then ping-pong from there.
    public func registerUndo(_ inverse: Command) {
        undoStack.append(inverse)
        if undoStack.count > limit { undoStack.removeFirst(undoStack.count - limit) }
        redoStack.removeAll()
    }
}

// MARK: - Concrete commands

public struct SetParam: Command {
    public let nodeID: UUID
    public let key: String
    public let value: NodeValue?   // nil = remove the key (so undo can reach "unset")
    public let mergeKey: String?

    public init(nodeID: UUID, key: String, value: NodeValue?, mergeKey: String? = nil) {
        self.nodeID = nodeID; self.key = key; self.value = value; self.mergeKey = mergeKey
    }

    public var label: String { "Set \(key)" }

    public func perform(_ doc: Document) -> Command {
        let old = doc.graph.node(nodeID)?.params[key]
        doc.graph.setParam(nodeID, key, value)
        doc.changes.didChange(nodeID)
        doc.markDirty()
        return SetParam(nodeID: nodeID, key: key, value: old, mergeKey: mergeKey)
    }
}

public struct MoveNode: Command {
    public let nodeID: UUID
    public let position: CGPoint
    public let mergeKey: String?

    public init(nodeID: UUID, position: CGPoint, mergeKey: String? = nil) {
        self.nodeID = nodeID; self.position = position; self.mergeKey = mergeKey
    }

    public var label: String { "Move Node" }

    public func perform(_ doc: Document) -> Command {
        let old = doc.graph.node(nodeID)?.position ?? position
        doc.graph.setPosition(nodeID, position)
        doc.changes.didChange(nodeID)
        doc.markDirty()
        return MoveNode(nodeID: nodeID, position: old, mergeKey: mergeKey)
    }
}

public struct AddNode: Command {
    public let node: Node
    public init(node: Node) { self.node = node }
    public var label: String { "Add \(node.type)" }

    public func perform(_ doc: Document) -> Command {
        doc.graph.nodes.append(node)
        doc.changes.didChangeGraph()
        doc.markDirty()
        return DeleteNode(nodeID: node.id)
    }
}

public struct DeleteNode: Command {
    public let nodeID: UUID
    public init(nodeID: UUID) { self.nodeID = nodeID }
    public var label: String { "Delete Node" }

    public func perform(_ doc: Document) -> Command {
        let node = doc.graph.node(nodeID)
        let conns = doc.graph.connections.filter { $0.fromNode == nodeID || $0.toNode == nodeID }
        doc.graph.removeNode(nodeID)
        doc.selection.remove(nodeID)
        if doc.viewNodeID == nodeID { doc.viewNodeID = nil }
        doc.changes.didChangeGraph()
        doc.markDirty()
        return RestoreNode(node: node, connections: conns)
    }
}

/// Inverse of `DeleteNode`: re-adds the node and the edges that touched it.
public struct RestoreNode: Command {
    public let node: Node?
    public let connections: [Connection]
    public init(node: Node?, connections: [Connection]) {
        self.node = node; self.connections = connections
    }
    public var label: String { "Restore Node" }

    public func perform(_ doc: Document) -> Command {
        guard let node else { return NoOp() }
        doc.graph.nodes.append(node)
        doc.graph.connections.append(contentsOf: connections)
        doc.changes.didChangeGraph()
        doc.markDirty()
        return DeleteNode(nodeID: node.id)
    }
}

/// Add a wire, displacing any existing wire into the same input (single-input).
public struct Connect: Command {
    public let connection: Connection
    public init(connection: Connection) { self.connection = connection }
    public var label: String { "Connect" }

    public func perform(_ doc: Document) -> Command {
        let displaced = doc.graph.connections.filter {
            $0.toNode == connection.toNode && $0.toPort == connection.toPort
        }
        doc.graph.connections.removeAll {
            $0.toNode == connection.toNode && $0.toPort == connection.toPort
        }
        doc.graph.connections.append(connection)
        doc.changes.didChangeGraph()
        doc.markDirty()
        return EditConnections(remove: [connection], add: displaced)
    }
}

/// Remove some wires and add others; its own inverse swaps the two sets.
/// Used for plain disconnects (add empty) and as `Connect`'s inverse.
public struct EditConnections: Command {
    public let remove: [Connection]
    public let add: [Connection]
    public init(remove: [Connection], add: [Connection]) {
        self.remove = remove; self.add = add
    }
    public var label: String { "Edit Connections" }

    public func perform(_ doc: Document) -> Command {
        doc.graph.connections.removeAll { remove.contains($0) }
        doc.graph.connections.append(contentsOf: add)
        doc.changes.didChangeGraph()
        doc.markDirty()
        return EditConnections(remove: add, add: remove)
    }
}

public struct NoOp: Command {
    public init() {}
    public var label: String { "No-op" }
    public func perform(_ doc: Document) -> Command { self }
}
