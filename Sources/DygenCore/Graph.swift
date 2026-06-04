import Foundation
import CoreGraphics

/// A node in the pipeline graph. `type` keys into `NodeRegistry`; `params`
/// holds its parameter values; `position` is its node-editor location.
public struct Node: Identifiable, Codable, Equatable {
    public let id: UUID
    public var type: String
    public var position: CGPoint
    public var params: [String: NodeValue]

    public init(id: UUID = UUID(), type: String, position: CGPoint = .zero, params: [String: NodeValue] = [:]) {
        self.id = id
        self.type = type
        self.position = position
        self.params = params
    }
}

/// A directed edge from one node's output port to another node's input port.
public struct Connection: Codable, Equatable, Hashable, Identifiable {
    public var fromNode: UUID
    public var fromPort: String
    public var toNode: UUID
    public var toPort: String

    public var id: String { "\(fromNode.uuidString):\(fromPort)->\(toNode.uuidString):\(toPort)" }

    public init(fromNode: UUID, fromPort: String, toNode: UUID, toPort: String) {
        self.fromNode = fromNode
        self.fromPort = fromPort
        self.toNode = toNode
        self.toPort = toPort
    }
}

/// The whole node graph: the executable pipeline and the persisted document body.
public struct Graph: Codable, Equatable {
    public var nodes: [Node]
    public var connections: [Connection]

    public init(nodes: [Node] = [], connections: [Connection] = []) {
        self.nodes = nodes
        self.connections = connections
    }

    // MARK: Queries

    public func node(_ id: UUID) -> Node? { nodes.first { $0.id == id } }
    public func index(of id: UUID) -> Int? { nodes.firstIndex { $0.id == id } }

    /// The single connection feeding a given input port, if any.
    public func inputConnection(to node: UUID, port: String) -> Connection? {
        connections.first { $0.toNode == node && $0.toPort == port }
    }

    /// Would adding `source -> target` introduce a cycle? True if `target`
    /// can already reach `source` by following existing edges forward.
    public func wouldCreateCycle(source: UUID, target: UUID) -> Bool {
        if source == target { return true }
        var stack = [target]
        var seen = Set<UUID>()
        while let n = stack.popLast() {
            if n == source { return true }
            if !seen.insert(n).inserted { continue }
            for c in connections where c.fromNode == n { stack.append(c.toNode) }
        }
        return false
    }

    // MARK: Mutations

    /// Set (non-nil) or remove (nil) a parameter. The nil case lets `SetParam`
    /// faithfully undo back to "unset".
    public mutating func setParam(_ id: UUID, _ key: String, _ value: NodeValue?) {
        if let i = index(of: id) { nodes[i].params[key] = value }
    }

    public mutating func setPosition(_ id: UUID, _ p: CGPoint) {
        if let i = index(of: id) { nodes[i].position = p }
    }

    /// Remove a node and every connection touching it.
    public mutating func removeNode(_ id: UUID) {
        nodes.removeAll { $0.id == id }
        connections.removeAll { $0.fromNode == id || $0.toNode == id }
    }
}
