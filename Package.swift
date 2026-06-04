// swift-tools-version: 6.0
import PackageDescription

// Dygen — GPU node-based macOS rebuild of the painterly pipeline.
// See docs/DESIGN.md and docs/PLAN.md.
//
// Two reusable libraries that build/test headlessly via `swift test`:
//   WindowKit  — dockable window/layout shell (adapted from dray)
//   DygenCore  — Document model, node graph, executor, GPU ops
//
// The SwiftUI app itself lives in Dygen.xcodeproj (a single app target that
// consumes these two as a local package dependency). Keeping the libraries in
// SwiftPM means their logic stays unit-testable without launching the app.
//
// Swift 5 language mode for now: keeps the dray ports free of Swift 6
// strict-concurrency churn while we stand things up. We tighten later.
let package = Package(
    name: "Dygen",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "WindowKit", targets: ["WindowKit"]),
        .library(name: "DygenCore", targets: ["DygenCore"]),
    ],
    targets: [
        .target(name: "WindowKit"),
        .target(name: "DygenCore"),
        .testTarget(name: "WindowKitTests", dependencies: ["WindowKit"]),
        .testTarget(name: "DygenCoreTests", dependencies: ["DygenCore"]),
    ],
    swiftLanguageModes: [.v5]
)
