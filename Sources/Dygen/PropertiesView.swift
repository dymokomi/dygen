import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DygenCore

/// Inspector for the selected node, driven by its `NodeDescriptor.params`.
/// Edits go through the document: discrete controls run a `SetParam` command;
/// scrub drags apply live and register one undo step on release.
struct PropertiesView: View {
    @ObservedObject var document: Document

    var body: some View {
        if let id = document.selection.first,
           let node = document.graph.node(id),
           let desc = NodeRegistry.descriptor(node.type) {
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    Text(desc.title).font(.system(size: 13, weight: .semibold))
                    Text(node.type).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
                    Divider()
                    ForEach(desc.params, id: \.key) { spec in
                        ParamRow(document: document, nodeID: id, spec: spec)
                    }
                    Spacer(minLength: 0)
                }
                .padding(12)
            }
        } else {
            VStack {
                Text(document.selection.count > 1 ? "\(document.selection.count) nodes selected" : "Select a node")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

private struct ParamRow: View {
    @ObservedObject var document: Document
    let nodeID: UUID
    let spec: ParamSpec
    @State private var dragStart: NodeValue?

    private var value: NodeValue {
        document.graph.node(nodeID)?.params[spec.key] ?? spec.defaultValue
    }

    var body: some View {
        HStack(spacing: 8) {
            Text(spec.label).font(.system(size: 11)).foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)
            control
        }
    }

    @ViewBuilder private var control: some View {
        switch spec.ui {
        case .scrubFloat:
            ScrubField(value: floatBinding, precision: 3, onEditingChanged: editing)
        case .scrubInt:
            ScrubField(value: floatBinding, speed: 0.25, precision: 0, onEditingChanged: editing)
        case .toggle:
            Toggle("", isOn: boolBinding).labelsHidden()
            Spacer()
        case .color:
            ColorPicker("", selection: colorBinding).labelsHidden()
            Spacer()
        case .dropdown:
            Picker("", selection: stringBinding) {
                ForEach(spec.choices ?? [], id: \.self) { Text($0).tag($0) }
            }
            .labelsHidden().frame(maxWidth: .infinity)
        case .filePicker:
            filePicker
        }
    }

    private var filePicker: some View {
        HStack(spacing: 6) {
            Text(displayPath)
                .font(.system(size: 10, design: .monospaced))
                .lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("…") { pickFile() }.controlSize(.small)
        }
    }

    private var displayPath: String {
        let s = value.stringValue ?? ""
        return s.isEmpty ? "—" : (s as NSString).lastPathComponent
    }

    // MARK: bindings

    private var floatBinding: Binding<Double> {
        Binding(
            get: { Double(value.floatValue ?? 0) },
            set: { nv in
                let v: NodeValue = (spec.ui == .scrubInt) ? .int(Int(nv.rounded())) : .float(Float(nv))
                document.graph.setParam(nodeID, spec.key, v)   // live, no undo entry
                document.changes.didChange(nodeID)
                document.markDirty()
            }
        )
    }

    private func editing(_ active: Bool) {
        if active {
            dragStart = value
        } else if let start = dragStart {
            document.commands.registerUndo(SetParam(nodeID: nodeID, key: spec.key, value: start))
            dragStart = nil
        }
    }

    private var boolBinding: Binding<Bool> {
        Binding(get: { value.boolValue ?? false },
                set: { document.run(SetParam(nodeID: nodeID, key: spec.key, value: .bool($0))) })
    }

    private var stringBinding: Binding<String> {
        Binding(get: { value.stringValue ?? "" },
                set: { document.run(SetParam(nodeID: nodeID, key: spec.key, value: .string($0))) })
    }

    private var colorBinding: Binding<Color> {
        Binding(
            get: { let c = value.rgb ?? (0, 0, 0); return Color(red: Double(c.0), green: Double(c.1), blue: Double(c.2)) },
            set: { newColor in
                let ns = NSColor(newColor).usingColorSpace(.deviceRGB) ?? .black
                document.run(SetParam(nodeID: nodeID, key: spec.key,
                                      value: .color(Float(ns.redComponent), Float(ns.greenComponent), Float(ns.blueComponent))))
            }
        )
    }

    private func pickFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            document.run(SetParam(nodeID: nodeID, key: spec.key, value: .string(url.path)))
        }
    }
}
