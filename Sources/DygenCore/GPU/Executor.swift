import Metal

/// Evaluates the node graph on the GPU with per-node output caching and dirty
/// propagation. A node re-runs only when itself or an upstream changed.
public final class Executor {
    public let ctx: GPUContext
    private let opForType: (String) -> NodeOp?
    private var cache: [UUID: NodeImage] = [:]
    private var dirty: Set<UUID> = []

    public init(ctx: GPUContext, opForType: @escaping (String) -> NodeOp?) {
        self.ctx = ctx
        self.opForType = opForType
    }

    /// Drop all cached outputs (e.g. on graph-structure change).
    public func invalidateAll() {
        cache.removeAll()
        dirty.removeAll()
    }

    /// Mark a node and everything downstream as needing re-evaluation.
    public func markDirty(_ id: UUID, graph: Graph) {
        var stack = [id]
        while let n = stack.popLast() {
            if dirty.contains(n) { continue }
            dirty.insert(n)
            cache[n] = nil
            for c in graph.connections where c.fromNode == n { stack.append(c.toNode) }
        }
    }

    /// Evaluate a node, recursively pulling its inputs. Returns its output image
    /// (nil for sinks / unresolved nodes). Cached unless dirty.
    @discardableResult
    public func evaluate(_ id: UUID?, graph: Graph) throws -> NodeImage? {
        guard let id else { return nil }
        if !dirty.contains(id), let cached = cache[id] { return cached }
        guard let node = graph.node(id), let op = opForType(node.type) else { return nil }

        var inputs: [String: NodeImage] = [:]
        if let desc = NodeRegistry.descriptor(node.type) {
            for port in desc.inputs {
                if let conn = graph.inputConnection(to: id, port: port.name),
                   let img = try evaluate(conn.fromNode, graph: graph) {
                    inputs[port.name] = img
                }
            }
        }

        let out = try op.evaluate(inputs: inputs, params: node.params, ctx: ctx)
        if let out, !out.isEmpty { cache[id] = out }   // sinks aren't cached → always re-run
        dirty.remove(id)
        return out
    }
}
