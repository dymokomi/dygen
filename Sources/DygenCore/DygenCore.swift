import Foundation

/// DygenCore — the document model, node graph, execution engine, and GPU ops.
///
/// M0 placeholder. The `Document`/`Graph`/`Node` model + command bus land in
/// M2; the Metal execution engine in M4+.
public enum DygenCore {
    public static let name = "DygenCore"

    /// Called once at app launch: registers built-in node types and logs.
    public static func bootstrap() {
        NodeRegistry.registerBuiltins()
        AppLog.shared.log("\(name) ready")
    }
}
