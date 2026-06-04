import SwiftUI
import AppKit
import WindowKit
import DygenCore

/// Ensures the app shows a real, focused window even when launched as a plain
/// SPM executable (no .app bundle yet). Once we wrap in a bundle this is moot.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

struct DygenApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = AppModel()

    init() {
        WindowKit.bootstrap()
        DygenCore.bootstrap()
    }

    var body: some Scene {
        WindowGroup("Dygen") {
            ContentView(model: model)
        }
        .defaultSize(width: 1200, height: 760)
        .commands {
            DygenCommands(model: model)
        }
    }
}
