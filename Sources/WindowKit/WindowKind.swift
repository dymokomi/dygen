import Foundation

/// Identifies a kind of dockable editor (canvas, node editor, properties, …).
///
/// WindowKit stays app-agnostic by treating a window kind as an opaque string
/// id. Only the id is persisted in the layout file; human-facing metadata
/// (title, icon) is resolved at render time via `WindowKindRegistry`. This is
/// the generic stand-in for what dray hardcodes as its `EditorType` enum.
public struct WindowKind: Codable, Hashable, Identifiable {
    public let id: String
    public init(_ id: String) { self.id = id }

    /// Fallback used only if an area somehow has no tabs (shouldn't happen).
    public static let placeholder = WindowKind("placeholder")
}

/// Display metadata for a window kind, supplied by the app.
public struct WindowKindInfo: Identifiable {
    public var id: String { kind.id }
    public let kind: WindowKind
    public let title: String
    public let systemImage: String

    public init(_ kind: WindowKind, title: String, systemImage: String) {
        self.kind = kind
        self.title = title
        self.systemImage = systemImage
    }
}

/// Maps window kinds to their display metadata, and lists the kinds offered in
/// the "add tab" / "set editor" menus. The app builds this once at launch.
public final class WindowKindRegistry: ObservableObject {
    public let infos: [WindowKindInfo]
    private let byID: [String: WindowKindInfo]

    public init(_ infos: [WindowKindInfo]) {
        self.infos = infos
        self.byID = Dictionary(infos.map { ($0.kind.id, $0) }, uniquingKeysWith: { a, _ in a })
    }

    public func info(for kind: WindowKind) -> WindowKindInfo? { byID[kind.id] }
    public func title(for kind: WindowKind) -> String { byID[kind.id]?.title ?? kind.id }
    public func systemImage(for kind: WindowKind) -> String { byID[kind.id]?.systemImage ?? "questionmark.square" }
}
