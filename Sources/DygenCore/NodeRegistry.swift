import Foundation
import CoreGraphics

/// What flows along a wire. v1 is image-only (per design C3); the painterly
/// nodes recompute the palette internally instead of carrying a typed port.
public enum PortKind: String, Codable, Equatable {
    case image
}

public struct PortSpec: Equatable {
    public let name: String
    public let kind: PortKind
    public init(_ name: String, _ kind: PortKind = .image) {
        self.name = name
        self.kind = kind
    }
}

/// How a parameter is edited in the Properties inspector.
public enum ParamUI: Equatable {
    case scrubFloat, scrubInt, toggle, color, filePicker, dropdown
}

public struct ParamSpec {
    public let key: String
    public let label: String
    public let ui: ParamUI
    public let defaultValue: NodeValue
    public let range: ClosedRange<Float>?
    public let choices: [String]?

    public init(_ key: String, label: String, ui: ParamUI, default defaultValue: NodeValue,
                range: ClosedRange<Float>? = nil, choices: [String]? = nil) {
        self.key = key
        self.label = label
        self.ui = ui
        self.defaultValue = defaultValue
        self.range = range
        self.choices = choices
    }
}

/// One description per node type drives the node editor (ports), the inspector
/// (params), and — from M4 — the executor (the GPU op). This is dray's
/// schema-driven `PropertyTemplate` idea unified with the graph.
public struct NodeDescriptor {
    public let type: String
    public let category: String
    public let title: String
    public let inputs: [PortSpec]
    public let outputs: [PortSpec]
    public let params: [ParamSpec]

    public init(type: String, category: String, title: String,
                inputs: [PortSpec], outputs: [PortSpec], params: [ParamSpec]) {
        self.type = type
        self.category = category
        self.title = title
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
    }

    /// A fresh `Node` of this type, params seeded from defaults.
    public func makeNode(at position: CGPoint = .zero) -> Node {
        var values: [String: NodeValue] = [:]
        for p in params { values[p.key] = p.defaultValue }
        return Node(type: type, position: position, params: values)
    }
}

/// Process-wide registry of node types. Populated at launch via `registerBuiltins()`.
public enum NodeRegistry {
    private static var store: [String: NodeDescriptor] = [:]

    public static func register(_ d: NodeDescriptor) { store[d.type] = d }
    public static func descriptor(_ type: String) -> NodeDescriptor? { store[type] }
    public static var all: [NodeDescriptor] { store.values.sorted { $0.title < $1.title } }

    /// Idempotent; safe to call more than once.
    public static func registerBuiltins() {
        register(NodeDescriptor(
            type: "read", category: "IO", title: "Read",
            inputs: [], outputs: [PortSpec("out")],
            params: [ParamSpec("path", label: "File", ui: .filePicker, default: .string(""))]
        ))
        register(NodeDescriptor(
            type: "write", category: "IO", title: "Write",
            inputs: [PortSpec("in")], outputs: [],
            params: [ParamSpec("path", label: "File", ui: .filePicker, default: .string(""))]
        ))

        // Filters (M5)
        register(NodeDescriptor(
            type: "pixelate", category: "Filter", title: "Pixelate",
            inputs: [PortSpec("in")], outputs: [PortSpec("out")],
            params: [ParamSpec("size", label: "Block", ui: .scrubInt, default: .int(2), range: 1...64)]
        ))
        register(NodeDescriptor(
            type: "blur", category: "Filter", title: "Blur",
            inputs: [PortSpec("in")], outputs: [PortSpec("out")],
            params: [ParamSpec("radius", label: "Radius", ui: .scrubFloat, default: .float(4), range: 0...64)]
        ))
        register(NodeDescriptor(
            type: "sharpen", category: "Filter", title: "Sharpen",
            inputs: [PortSpec("in")], outputs: [PortSpec("out")],
            params: []
        ))

        // Quantize family (M6)
        register(NodeDescriptor(
            type: "quantize", category: "Quantize", title: "Quantize",
            inputs: [PortSpec("in")], outputs: [PortSpec("out")],
            params: [ParamSpec("colors", label: "Colors", ui: .scrubInt, default: .int(64), range: 2...64)]
        ))
        register(NodeDescriptor(
            type: "cleanup", category: "Quantize", title: "Cleanup",
            inputs: [PortSpec("in")], outputs: [PortSpec("out")],
            params: [ParamSpec("radius", label: "Cluster", ui: .scrubInt, default: .int(13), range: 0...40)]
        ))

        // Painterly (M7)
        register(NodeDescriptor(
            type: "paintBlobs", category: "Paint", title: "Paint Blobs",
            inputs: [PortSpec("in")], outputs: [PortSpec("out")],
            params: [ParamSpec("seed", label: "Seed", ui: .scrubInt, default: .int(2), range: 0...999)]
        ))
        register(NodeDescriptor(
            type: "brushStrokes", category: "Paint", title: "Brush Strokes",
            inputs: [PortSpec("in"), PortSpec("stencil")], outputs: [PortSpec("out")],
            params: [
                ParamSpec("amount", label: "Amount", ui: .scrubFloat, default: .float(0.5), range: 0...1),
                ParamSpec("softness", label: "Softness", ui: .scrubFloat, default: .float(12), range: 0...40),
                ParamSpec("volumeDiff", label: "Vol Diff", ui: .scrubFloat, default: .float(0.03), range: 0...1),
                ParamSpec("seed", label: "Seed", ui: .scrubInt, default: .int(2), range: 0...999),
            ]
        ))
        register(NodeDescriptor(
            type: "compOriginal", category: "Paint", title: "Comp Original",
            inputs: [PortSpec("in"), PortSpec("original"), PortSpec("stencil")], outputs: [PortSpec("out")],
            params: []
        ))
        register(NodeDescriptor(
            type: "glow", category: "Filter", title: "Glow",
            inputs: [PortSpec("in")], outputs: [PortSpec("out")],
            params: [
                ParamSpec("radius", label: "Radius", ui: .scrubFloat, default: .float(15), range: 0...64),
                ParamSpec("amount", label: "Amount", ui: .scrubFloat, default: .float(0.3), range: 0...1),
            ]
        ))
    }
}
