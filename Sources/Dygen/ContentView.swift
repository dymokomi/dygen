import SwiftUI
import WindowKit
import DygenCore

/// Hosts the WindowKit docking surface over the app model. The layout manager
/// owns the live split tree (restored on appear, auto-persisted).
struct ContentView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        DockView(manager: model.layout) { kind, _ in
            DygenWindows.editor(for: kind, model: model)
        }
        .frame(minWidth: 900, minHeight: 560)
        .onAppear { model.layout.restoreLayout() }
    }
}
