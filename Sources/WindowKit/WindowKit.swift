import Foundation

/// WindowKit — a reusable, dockable window/layout shell.
///
/// M0 placeholder. The docking model (binary split tree, `LayoutManager`,
/// recursive area renderer, dividers) lands in M1, adapted from dray.
public enum WindowKit {
    public static let name = "WindowKit"

    /// Called once at app launch to prove the module links and is reachable.
    public static func bootstrap() {
        print("[\(name)] ready")
    }
}
