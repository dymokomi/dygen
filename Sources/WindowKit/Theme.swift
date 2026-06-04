import SwiftUI

/// Neutral dark palette for the docking chrome. Apps may restyle later; for now
/// these match the editor-grey look the layout was designed against.
public enum Theme {
    public static let bg = Color(white: 0x32 / 255.0)
    public static let inputBg = Color(white: 0x29 / 255.0)
    public static let separator = Color(white: 0x1A / 255.0)
    public static let selected = Color(white: 0x4A / 255.0)
    public static let hoverBg = Color(white: 0x3C / 255.0)
    public static let textPrimary = Color(white: 0.85)
    public static let textSecondary = Color(white: 0.55)
}
