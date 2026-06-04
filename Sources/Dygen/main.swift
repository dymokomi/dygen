import Foundation
import DygenCore

// Entry point: `--headless` runs the CLI driver (using the bundled default
// scene when no document is opened); otherwise launch the SwiftUI app.
if CommandLine.arguments.contains("--headless") {
    let code = Headless.run(Array(CommandLine.arguments.dropFirst())) {
        let scene = DefaultScene.make()
        return (scene.graph, scene.viewNode)
    }
    exit(code)
} else {
    DygenApp.main()
}
