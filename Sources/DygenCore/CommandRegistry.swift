import Foundation
import CoreGraphics

/// Context for building a command from serialized params. Resolves node
/// references by UUID string or by node type (first node of that type) — the
/// latter is what makes headless `--set quantize colors 32` ergonomic.
public struct CommandContext {
    public let document: Document
    public init(document: Document) { self.document = document }

    public func node(_ ref: String) -> UUID? {
        if let u = UUID(uuidString: ref) { return document.graph.node(u) != nil ? u : nil }
        return document.graph.nodes.first { $0.type == ref }?.id
    }
}

/// Codable command params + a factory producing the runtime `Command`. This is
/// the public surface for scripted/headless/RPC dispatch (mirrors dray).
public protocol RegisterableCommand: Codable {
    static var commandName: String { get }
    func makeCommand(context: CommandContext) throws -> Command
}

public enum CommandRegistryError: Error, CustomStringConvertible {
    case unknown(String)
    case badParams(String, Error)
    case build(String, Error)

    public var description: String {
        switch self {
        case .unknown(let n):       return "Unknown command '\(n)'"
        case .badParams(let n, let e): return "Bad params for '\(n)': \(e)"
        case .build(let n, let e):  return "Could not build '\(n)': \(e)"
        }
    }
}

/// Looks up commands by name, decodes their params from JSON, and dispatches.
public final class CommandRegistry {
    public static let shared = CommandRegistry()

    private var decoders: [String: (Data, CommandContext) throws -> Command] = [:]
    public var names: [String] { decoders.keys.sorted() }

    public func register<C: RegisterableCommand>(_ type: C.Type) {
        decoders[C.commandName] = { data, ctx in
            let params: C
            do { params = try JSONDecoder().decode(C.self, from: data) }
            catch { throw CommandRegistryError.badParams(C.commandName, error) }
            do { return try params.makeCommand(context: ctx) }
            catch { throw CommandRegistryError.build(C.commandName, error) }
        }
    }

    public func make(name: String, paramsJSON: Data, context: CommandContext) throws -> Command {
        guard let dec = decoders[name] else { throw CommandRegistryError.unknown(name) }
        return try dec(paramsJSON, context)
    }

    /// Build from name + JSON params and run it on the document.
    @discardableResult
    public func dispatch(name: String, paramsJSON: Data, on doc: Document) throws -> Command {
        let cmd = try make(name: name, paramsJSON: paramsJSON, context: CommandContext(document: doc))
        doc.run(cmd)
        return cmd
    }

    public func registerBuiltins() {
        register(SetParamCommand.self)
        register(AddNodeCommand.self)
        register(DeleteNodeCommand.self)
        register(ConnectCommand.self)
        register(SetViewCommand.self)
    }
}

// MARK: - Built-in registerable commands

public struct SetParamCommand: RegisterableCommand {
    public static let commandName = "setParam"
    public var node: String
    public var key: String
    public var value: NodeValue
    public init(node: String, key: String, value: NodeValue) { self.node = node; self.key = key; self.value = value }
    public func makeCommand(context: CommandContext) throws -> Command {
        guard let id = context.node(node) else { throw NodeOpError.io("node '\(node)' not found") }
        return SetParam(nodeID: id, key: key, value: value)
    }
}

public struct AddNodeCommand: RegisterableCommand {
    public static let commandName = "addNode"
    public var type: String
    public var x: Double
    public var y: Double
    public init(type: String, x: Double = 0, y: Double = 0) { self.type = type; self.x = x; self.y = y }
    public func makeCommand(context: CommandContext) throws -> Command {
        guard let d = NodeRegistry.descriptor(type) else { throw NodeOpError.io("unknown node type '\(type)'") }
        return AddNode(node: d.makeNode(at: CGPoint(x: x, y: y)))
    }
}

public struct DeleteNodeCommand: RegisterableCommand {
    public static let commandName = "deleteNode"
    public var node: String
    public init(node: String) { self.node = node }
    public func makeCommand(context: CommandContext) throws -> Command {
        guard let id = context.node(node) else { throw NodeOpError.io("node '\(node)' not found") }
        return DeleteNode(nodeID: id)
    }
}

public struct ConnectCommand: RegisterableCommand {
    public static let commandName = "connect"
    public var from: String
    public var fromPort: String
    public var to: String
    public var toPort: String
    public init(from: String, fromPort: String, to: String, toPort: String) {
        self.from = from; self.fromPort = fromPort; self.to = to; self.toPort = toPort
    }
    public func makeCommand(context: CommandContext) throws -> Command {
        guard let f = context.node(from), let t = context.node(to) else { throw NodeOpError.io("node not found") }
        return Connect(connection: Connection(fromNode: f, fromPort: fromPort, toNode: t, toPort: toPort))
    }
}

/// Set which node the Canvas views (not undoable; applies immediately).
public struct SetViewCommand: RegisterableCommand {
    public static let commandName = "setView"
    public var node: String
    public init(node: String) { self.node = node }
    public func makeCommand(context: CommandContext) throws -> Command {
        guard let id = context.node(node) else { throw NodeOpError.io("node '\(node)' not found") }
        return SetViewNodeCommand(nodeID: id)
    }
}

/// Lightweight non-undoable command that sets the document's view node.
public struct SetViewNodeCommand: Command {
    public let nodeID: UUID
    public var label: String { "Set View" }
    public func perform(_ doc: Document) -> Command {
        let old = doc.viewNodeID
        doc.viewNodeID = nodeID
        return SetViewNodeCommand(nodeID: old ?? nodeID)
    }
}
