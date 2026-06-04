import Foundation
import CoreGraphics

/// Pan/zoom transform for the node editor's 2D viewport. Pure geometry, so it
/// lives in DygenCore and is unit-testable. Ported from dray's NodeGraphViewport.
public struct NodeGraphViewport: Equatable {
    public var zoom: CGFloat
    public var panX: CGFloat
    public var panY: CGFloat
    public var width: CGFloat
    public var height: CGFloat

    public init(zoom: CGFloat, panX: CGFloat, panY: CGFloat, width: CGFloat, height: CGFloat) {
        self.zoom = zoom; self.panX = panX; self.panY = panY; self.width = width; self.height = height
    }

    public func screenToWorld(_ sx: CGFloat, _ sy: CGFloat) -> CGPoint {
        CGPoint(x: (sx - width / 2 - panX) / zoom,
                y: (sy - height / 2 - panY) / zoom)
    }

    public func worldToScreen(_ wx: CGFloat, _ wy: CGFloat) -> CGPoint {
        CGPoint(x: wx * zoom + width / 2 + panX,
                y: wy * zoom + height / 2 + panY)
    }

    /// Zoom centered on a screen point, clamped. The world point under
    /// `screenPoint` is invariant before/after.
    public mutating func applyZoom(_ newZoomRaw: CGFloat, around screenPoint: CGPoint,
                                   minZoom: CGFloat = 0.1, maxZoom: CGFloat = 4.0) {
        let newZoom = min(max(newZoomRaw, minZoom), maxZoom)
        let before = screenToWorld(screenPoint.x, screenPoint.y)
        zoom = newZoom
        let after = screenToWorld(screenPoint.x, screenPoint.y)
        panX += (after.x - before.x) * zoom
        panY += (after.y - before.y) * zoom
    }
}
