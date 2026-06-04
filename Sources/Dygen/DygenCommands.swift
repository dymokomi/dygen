import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DygenCore

/// File, Edit, and Render menu commands operating on the shared document.
struct DygenCommands: Commands {
    let model: AppModel
    private var document: Document { model.document }
    private static let dygenType = UTType(filenameExtension: "dygen") ?? .json

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New") { document.reset() }.keyboardShortcut("n")
            Button("Open…") { open() }.keyboardShortcut("o")
            Divider()
            Button("Save") { save() }.keyboardShortcut("s")
            Button("Save As…") { saveAs() }.keyboardShortcut("s", modifiers: [.command, .shift])
        }
        CommandGroup(replacing: .undoRedo) {
            Button("Undo") { document.undo() }.keyboardShortcut("z")
            Button("Redo") { document.redo() }.keyboardShortcut("z", modifiers: [.command, .shift])
        }
        CommandMenu("Render") {
            Button("Render Write Nodes") { model.render() }.keyboardShortcut("r")
        }
    }

    private func open() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [Self.dygenType]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url,
              let loaded = try? Document.load(from: url) else { return }
        document.replaceContents(graph: loaded.graph, viewNode: loaded.viewNodeID, fileURL: url)
        AppLog.shared.log("Opened \(url.lastPathComponent)")
    }

    private func save() {
        if let url = document.fileURL {
            try? document.save(to: url)
            AppLog.shared.log("Saved \(url.lastPathComponent)")
        } else {
            saveAs()
        }
    }

    private func saveAs() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [Self.dygenType]
        panel.nameFieldStringValue = "Untitled.dygen"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? document.save(to: url)
        AppLog.shared.log("Saved \(url.lastPathComponent)")
    }
}
