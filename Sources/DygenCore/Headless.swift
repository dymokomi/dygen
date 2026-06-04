import Foundation

/// Headless driver: runs the pipeline and command dispatch without a GUI, for
/// CLI use (`Dygen --headless …`) and headless tests.
///
/// Flags (processed left→right):
///   --open <file.dygen>         load a document
///   --set <node> <key> <value>  setParam (node = type or UUID)
///   --cmd <name> <jsonParams>   dispatch any registered command
///   --view <node>               set the view node
///   --render                    evaluate all Write nodes (writes their paths)
///   --export <file.png>         render the view node to a PNG
///   --save <file.dygen>         save the document
public enum Headless {
    @discardableResult
    public static func run(_ args: [String], defaultGraph: (() -> (graph: Graph, viewNode: UUID?))? = nil) -> Int32 {
        NodeRegistry.registerBuiltins()
        CommandRegistry.shared.registerBuiltins()
        guard let gpu = GPUContext() else { err("no Metal device"); return 1 }
        let ex = Executor(ctx: gpu, opForType: BuiltinOps.op(for:))
        let doc = Document()
        if let dg = defaultGraph {
            let g = dg(); doc.graph = g.graph; doc.viewNodeID = g.viewNode
        }

        var status: Int32 = 0
        var i = 0
        func arg(_ off: Int) -> String? { (i + off) < args.count ? args[i + off] : nil }

        while i < args.count {
            switch args[i] {
            case "--headless":
                i += 1
            case "--open":
                guard let p = arg(1) else { err("--open needs a path"); return 2 }
                if let loaded = try? Document.load(from: URL(fileURLWithPath: p)) {
                    doc.replaceContents(graph: loaded.graph, viewNode: loaded.viewNodeID, fileURL: URL(fileURLWithPath: p))
                    log("opened \(p)")
                } else { err("could not open \(p)"); status = 3 }
                ex.invalidateAll(); i += 2
            case "--set":
                guard let n = arg(1), let k = arg(2), let v = arg(3) else { err("--set needs <node> <key> <value>"); return 2 }
                dispatch("setParam", SetParamCommand(node: n, key: k, value: parse(v)), doc)
                ex.invalidateAll(); i += 4
            case "--cmd":
                guard let name = arg(1), let json = arg(2) else { err("--cmd needs <name> <json>"); return 2 }
                do { try CommandRegistry.shared.dispatch(name: name, paramsJSON: Data(json.utf8), on: doc); log("ran \(name)") }
                catch { err("\(error)"); status = 4 }
                ex.invalidateAll(); i += 3
            case "--view":
                guard let n = arg(1) else { err("--view needs <node>"); return 2 }
                dispatch("setView", SetViewCommand(node: n), doc); i += 2
            case "--render":
                var c = 0
                for n in doc.graph.nodes where n.type == "write" { _ = try? ex.evaluate(n.id, graph: doc.graph); c += 1 }
                log("rendered \(c) write node(s)"); i += 1
            case "--export":
                guard let p = arg(1) else { err("--export needs a path"); return 2 }
                let vn = doc.viewNodeID ?? doc.graph.nodes.last?.id
                if let vn, let tex = try? ex.evaluate(vn, graph: doc.graph)?.texture {
                    try? TextureIO.writePNG(tex, to: URL(fileURLWithPath: p), ctx: gpu); log("exported \(p)")
                } else { err("nothing to export"); status = 5 }
                i += 2
            case "--save":
                guard let p = arg(1) else { err("--save needs a path"); return 2 }
                try? doc.save(to: URL(fileURLWithPath: p)); log("saved \(p)"); i += 2
            default:
                i += 1
            }
        }
        return status
    }

    private static func dispatch<C: RegisterableCommand>(_ name: String, _ params: C, _ doc: Document) {
        guard let json = try? JSONEncoder().encode(params) else { err("encode \(name) failed"); return }
        do { try CommandRegistry.shared.dispatch(name: name, paramsJSON: json, on: doc) }
        catch { err("\(error)") }
    }

    static func parse(_ s: String) -> NodeValue {
        if let i = Int(s) { return .int(i) }
        if let f = Float(s) { return .float(f) }
        if s == "true" { return .bool(true) }
        if s == "false" { return .bool(false) }
        return .string(s)
    }

    static func log(_ m: String) { AppLog.shared.log("[headless] \(m)") }
    static func err(_ m: String) { AppLog.shared.error("[headless] \(m)") }
}
