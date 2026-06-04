import SwiftUI
import WindowKit
import DygenCore

/// Top-level app state: the shared document, dock layout, GPU context, and the
/// executor. Subscribes the executor to document changes for dirty tracking.
final class AppModel: ObservableObject {
    let document: Document
    let layout: LayoutManager
    let gpu: GPUContext?
    let executor: Executor?

    init() {
        let doc = DygenWindows.makeDocument()
        document = doc
        layout = DygenWindows.makeLayoutManager()
        let gpu = GPUContext()
        self.gpu = gpu
        let ex = gpu.map { Executor(ctx: $0, opForType: BuiltinOps.op(for:)) }
        executor = ex

        doc.changes.subscribe { [weak ex, weak doc] change in
            guard let doc else { return }
            switch change {
            case .node(let id): ex?.markDirty(id, graph: doc.graph)
            case .graph:        ex?.invalidateAll()
            }
        }

        if gpu == nil { AppLog.shared.error("No Metal device available — GPU disabled") }
    }

    /// Evaluate all Write nodes (their side effect is writing a PNG).
    func render() {
        guard let ex = executor else { AppLog.shared.error("Render: no GPU"); return }
        let g = document.graph
        var count = 0
        for n in g.nodes where n.type == "write" {
            do { try ex.evaluate(n.id, graph: g); count += 1 }
            catch { AppLog.shared.error("Render: \(error)") }
        }
        AppLog.shared.log("Rendered \(count) write node(s)")
    }

    /// Export the current view node's output to a PNG at full resolution.
    func exportPNG(to url: URL) {
        guard let ex = executor, let vn = document.viewNodeID else { AppLog.shared.error("Export: nothing to view"); return }
        do {
            guard let tex = try ex.evaluate(vn, graph: document.graph)?.texture else {
                AppLog.shared.error("Export: view node produced no image"); return
            }
            try TextureIO.writePNG(tex, to: url, ctx: ex.ctx)
            AppLog.shared.log("Exported \(url.lastPathComponent)")
        } catch {
            AppLog.shared.error("Export: \(error)")
        }
    }
}
