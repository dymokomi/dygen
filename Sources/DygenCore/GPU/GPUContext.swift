import Metal

/// Shared Metal device + command queue. (Texture pooling arrives in M5+; M4 ops
/// allocate directly.) Failable: returns nil when no Metal device is available.
public final class GPUContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    public init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        device = d
        queue = q
    }
}
