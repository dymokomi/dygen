import Metal

/// The payload that flows between nodes. The graph model has image-only ports;
/// internally a node output is an RGBA8 `texture` plus, for the quantize family,
/// an optional palette-`indexTexture` (R8Uint) + `palette` buffer (RGBA float4
/// per entry). Downstream painterly nodes consume these without a second port type.
public struct NodeImage {
    public let texture: MTLTexture?
    public let indexTexture: MTLTexture?
    public let palette: MTLBuffer?
    public let paletteCount: Int

    public init(texture: MTLTexture?, indexTexture: MTLTexture? = nil,
                palette: MTLBuffer? = nil, paletteCount: Int = 0) {
        self.texture = texture
        self.indexTexture = indexTexture
        self.palette = palette
        self.paletteCount = paletteCount
    }

    public var width: Int { texture?.width ?? indexTexture?.width ?? 0 }
    public var height: Int { texture?.height ?? indexTexture?.height ?? 0 }
    public var isEmpty: Bool { texture == nil && indexTexture == nil }
}

/// A node's GPU operation: input images + params → output image (nil = sink).
public protocol NodeOp {
    func evaluate(inputs: [String: NodeImage], params: [String: NodeValue], ctx: GPUContext) throws -> NodeImage?
}

public enum NodeOpError: Error {
    case missingInput(String)
    case missingParam(String)
    case io(String)
}
