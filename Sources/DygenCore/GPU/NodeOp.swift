import Metal

/// A node's GPU operation: take input textures + params, produce an output
/// texture (or nil for a sink like Write). From M5 these encode compute kernels.
public protocol NodeOp {
    func evaluate(inputs: [String: MTLTexture], params: [String: NodeValue], ctx: GPUContext) throws -> MTLTexture?
}

public enum NodeOpError: Error {
    case missingInput(String)
    case missingParam(String)
    case io(String)
}
