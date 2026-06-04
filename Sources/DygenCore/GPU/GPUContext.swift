import Metal
import Foundation

/// Shared Metal device, command queue, and the runtime-compiled kernel library.
/// Failable: returns nil when no Metal device is available or kernels fail to
/// compile (the MSL error is written to stderr so it's debuggable).
public final class GPUContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    let library: MTLLibrary

    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    public init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        do {
            library = try d.makeLibrary(source: KernelSource.all, options: nil)
        } catch {
            FileHandle.standardError.write(Data("[GPUContext] kernel compile failed: \(error)\n".utf8))
            return nil
        }
        device = d
        queue = q
    }

    // MARK: - Pipelines

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) else {
            throw NodeOpError.io("kernel '\(name)' not found")
        }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }

    // MARK: - Textures

    /// A processing texture (read/write/render, private storage).
    public func makeTexture(width: Int, height: Int, pixelFormat: MTLPixelFormat = .rgba8Unorm) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat, width: max(1, width), height: max(1, height), mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite, .renderTarget]
        d.storageMode = .private
        return device.makeTexture(descriptor: d)
    }

    // MARK: - Dispatch

    /// Encode a compute kernel over `output`'s extent. Output is texture index 0,
    /// inputs follow at index 1.... `setup` sets any constant bytes/buffers.
    func encode(kernel name: String, output: MTLTexture, inputs: [MTLTexture],
                buffers: [MTLBuffer?] = [], setup: ((MTLComputeCommandEncoder) -> Void)? = nil,
                in cb: MTLCommandBuffer) throws {
        guard let enc = cb.makeComputeCommandEncoder() else { throw NodeOpError.io("no compute encoder") }
        let pso = try pipeline(name)
        enc.setComputePipelineState(pso)
        enc.setTexture(output, index: 0)
        for (i, t) in inputs.enumerated() { enc.setTexture(t, index: i + 1) }
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        setup?(enc)
        let w = pso.threadExecutionWidth
        let h = max(1, pso.maxTotalThreadsPerThreadgroup / w)
        enc.dispatchThreads(MTLSize(width: output.width, height: output.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        enc.endEncoding()
    }

    /// Convenience: run a single kernel in its own committed command buffer.
    @discardableResult
    func runKernel(_ name: String, output: MTLTexture, inputs: [MTLTexture],
                   setup: ((MTLComputeCommandEncoder) -> Void)? = nil) throws -> MTLTexture {
        guard let cb = queue.makeCommandBuffer() else { throw NodeOpError.io("no command buffer") }
        try encode(kernel: name, output: output, inputs: inputs, setup: setup, in: cb)
        cb.commit()
        return output
    }
}
