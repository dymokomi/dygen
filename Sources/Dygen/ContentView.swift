import SwiftUI
import WindowKit
import DygenCore

/// Hosts the WindowKit docking surface over the shared `Document`. The layout
/// manager owns the live split tree (restored on appear, auto-persisted).
struct ContentView: View {
    @ObservedObject var document: Document
    @ObservedObject var layout: LayoutManager

    var body: some View {
        DockView(manager: layout) { kind, _ in
            DygenWindows.editor(for: kind, document: document)
        }
        .frame(minWidth: 900, minHeight: 560)
        .onAppear { layout.restoreLayout() }
    }
}
