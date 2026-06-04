import AppKit
import CoreGraphics
import DygenCore

/// AppKit + CoreGraphics node-graph canvas, ported and de-USD'd from dray's
/// NodeGraphCanvasView. Drives off our `Node` / `NodeDescriptor` model and emits
/// `Command`s. Drawing is done in screen space (sizes scaled by zoom) to avoid
/// flipped-context text orientation issues.
final class NodeCanvasView: NSView {

    // Layout constants
    enum L {
        static let nodeWidth: CGFloat = 150
        static let headerHeight: CGFloat = 22
        static let portHeight: CGFloat = 18
        static let portRadius: CGFloat = 4.5
        static let minHeight: CGFloat = 46
        static let cpOffset: CGFloat = 60
    }

    // Callbacks to the SwiftUI layer
    var onRun: ((Command) -> Void)?
    var onSelect: ((Set<UUID>) -> Void)?
    var onSetViewNode: ((UUID) -> Void)?
    var descriptorFor: (String) -> NodeDescriptor? = { NodeRegistry.descriptor($0) }

    // Model snapshot
    private var nodes: [Node] = []
    private var connections: [Connection] = []
    private(set) var selected: Set<UUID> = []
    private var viewNodeID: UUID?
    private var positions: [UUID: CGPoint] = [:]

    // Viewport
    private var zoom: CGFloat = 1
    private var panX: CGFloat = 0
    private var panY: CGFloat = 0
    static let minZoom: CGFloat = 0.15
    static let maxZoom: CGFloat = 3.0

    // Interaction state
    private var isDraggingNode = false
    private var draggedNodeID: UUID?
    private var multiDragOffsets: [UUID: CGPoint] = [:]
    private var dragOffset: CGPoint = .zero
    private var isMiddleDragging = false
    private var isBoxSelecting = false
    private var boxStart: CGPoint = .zero
    private var boxEnd: CGPoint = .zero
    private var isCutting = false
    private var cutStroke: [CGPoint] = []
    private var draggingConnection: (node: UUID, port: String, isOutput: Bool)?
    private var connectionDragPoint: CGPoint = .zero
    private var hoverDrop: (node: UUID, port: String, isOutput: Bool)?
    private var lastMouse: CGPoint = .zero
    private var didDragNode = false
    private var pendingAddPos: CGPoint = .zero

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    // MARK: - Model sync

    func apply(nodes: [Node], connections: [Connection], selected: Set<UUID>, viewNode: UUID?) {
        self.nodes = nodes
        self.connections = connections
        self.selected = selected
        self.viewNodeID = viewNode
        if !isDraggingNode {
            for n in nodes { positions[n.id] = n.position }
        }
        let ids = Set(nodes.map(\.id))
        positions = positions.filter { ids.contains($0.key) }
        needsDisplay = true
    }

    // MARK: - Transforms

    private func viewport() -> NodeGraphViewport {
        NodeGraphViewport(zoom: zoom, panX: panX, panY: panY, width: bounds.width, height: bounds.height)
    }
    private func toWorld(_ p: CGPoint) -> CGPoint { viewport().screenToWorld(p.x, p.y) }
    private func toScreen(_ x: CGFloat, _ y: CGFloat) -> CGPoint { viewport().worldToScreen(x, y) }

    // MARK: - Geometry helpers

    private func ports(_ node: Node) -> (inputs: [String], outputs: [String]) {
        guard let d = descriptorFor(node.type) else { return ([], []) }
        return (d.inputs.map(\.name), d.outputs.map(\.name))
    }

    private func height(_ node: Node) -> CGFloat {
        let p = ports(node)
        let n = max(p.inputs.count, p.outputs.count, 1)
        return max(L.minHeight, L.headerHeight + CGFloat(n) * L.portHeight + 8)
    }

    private func node(_ id: UUID) -> Node? { nodes.first { $0.id == id } }

    /// World-space center of a port.
    private func portPos(_ node: Node, _ portName: String, isInput: Bool) -> CGPoint? {
        guard let pos = positions[node.id] else { return nil }
        let list = isInput ? ports(node).inputs : ports(node).outputs
        guard let idx = list.firstIndex(of: portName) else { return nil }
        let x = isInput ? pos.x : pos.x + L.nodeWidth
        let y = pos.y + L.headerHeight + CGFloat(idx) * L.portHeight + L.portHeight / 2
        return CGPoint(x: x, y: y)
    }

    // MARK: - Hit testing (world coords)

    private func nodeAt(_ w: CGPoint) -> Node? {
        for node in nodes.reversed() {
            guard let pos = positions[node.id] else { continue }
            let h = height(node)
            if w.x >= pos.x, w.x <= pos.x + L.nodeWidth, w.y >= pos.y, w.y <= pos.y + h { return node }
        }
        return nil
    }

    private func portAt(_ w: CGPoint) -> (node: UUID, port: String, isOutput: Bool)? {
        let r: CGFloat = 9
        for node in nodes {
            for p in ports(node).inputs {
                if let c = portPos(node, p, isInput: true) {
                    let dx = w.x - c.x, dy = w.y - c.y
                    if dx*dx + dy*dy < r*r { return (node.id, p, false) }
                }
            }
            for p in ports(node).outputs {
                if let c = portPos(node, p, isInput: false) {
                    let dx = w.x - c.x, dy = w.y - c.y
                    if dx*dx + dy*dy < r*r { return (node.id, p, true) }
                }
            }
        }
        return nil
    }

    private func bezierPoint(_ s: CGPoint, _ e: CGPoint, _ t: CGFloat) -> CGPoint {
        let cp1 = CGPoint(x: s.x + L.cpOffset, y: s.y), cp2 = CGPoint(x: e.x - L.cpOffset, y: e.y)
        let mt = 1 - t
        let x = mt*mt*mt*s.x + 3*mt*mt*t*cp1.x + 3*mt*t*t*cp2.x + t*t*t*e.x
        let y = mt*mt*mt*s.y + 3*mt*mt*t*cp1.y + 3*mt*t*t*cp2.y + t*t*t*e.y
        return CGPoint(x: x, y: y)
    }

    private func connectionEndpoints(_ c: Connection) -> (CGPoint, CGPoint)? {
        guard let s = node(c.fromNode), let e = node(c.toNode),
              let sp = portPos(s, c.fromPort, isInput: false),
              let ep = portPos(e, c.toPort, isInput: true) else { return nil }
        return (sp, ep)
    }

    private func edgeAt(_ w: CGPoint) -> Int? {
        let thr: CGFloat = 7
        for (i, c) in connections.enumerated() {
            guard let (s, e) = connectionEndpoints(c) else { continue }
            for step in 0...20 {
                let p = bezierPoint(s, e, CGFloat(step)/20)
                let dx = w.x - p.x, dy = w.y - p.y
                if dx*dx + dy*dy < thr*thr { return i }
            }
        }
        return nil
    }

    private func edgesHit(byStroke stroke: [CGPoint]) -> [Int] {
        guard stroke.count >= 2 else { return [] }
        var hits = Set<Int>()
        for (i, c) in connections.enumerated() {
            guard let (s, e) = connectionEndpoints(c) else { continue }
            var samples: [CGPoint] = []
            for step in 0...20 { samples.append(bezierPoint(s, e, CGFloat(step)/20)) }
            outer: for si in 0..<(samples.count-1) {
                for ki in 0..<(stroke.count-1) {
                    if segIntersect(samples[si], samples[si+1], stroke[ki], stroke[ki+1]) { hits.insert(i); break outer }
                }
            }
        }
        return Array(hits)
    }

    private func nodesIn(rect: CGRect) -> [UUID] {
        nodes.compactMap { n in
            guard let pos = positions[n.id] else { return nil }
            let r = CGRect(x: pos.x, y: pos.y, width: L.nodeWidth, height: height(n))
            return rect.intersects(r) ? n.id : nil
        }
    }

    private func segIntersect(_ p1: CGPoint, _ p2: CGPoint, _ p3: CGPoint, _ p4: CGPoint) -> Bool {
        func ccw(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> CGFloat { (c.y-a.y)*(b.x-a.x) - (b.y-a.y)*(c.x-a.x) }
        let d1 = ccw(p3,p4,p1), d2 = ccw(p3,p4,p2), d3 = ccw(p1,p2,p3), d4 = ccw(p1,p2,p4)
        return ((d1>0 && d2<0) || (d1<0 && d2>0)) && ((d3>0 && d4<0) || (d3<0 && d4>0))
    }

    private func worldRect(_ a: CGPoint, _ b: CGPoint) -> CGRect {
        CGRect(x: min(a.x,b.x), y: min(a.y,b.y), width: abs(a.x-b.x), height: abs(a.y-b.y))
    }

    private func isPortConnected(_ nodeID: UUID, _ port: String, isOutput: Bool) -> Bool {
        connections.contains { isOutput ? ($0.fromNode == nodeID && $0.fromPort == port)
                                        : ($0.toNode == nodeID && $0.toPort == port) }
    }

    private func headerColor(_ node: Node) -> NSColor {
        switch descriptorFor(node.type)?.category {
        case "IO":       return NSColor(calibratedHue: 0.58, saturation: 0.5, brightness: 0.55, alpha: 1)
        case "Quantize": return NSColor(calibratedHue: 0.78, saturation: 0.45, brightness: 0.55, alpha: 1)
        case "Paint":    return NSColor(calibratedHue: 0.08, saturation: 0.55, brightness: 0.6, alpha: 1)
        case "Filter":   return NSColor(calibratedHue: 0.45, saturation: 0.45, brightness: 0.5, alpha: 1)
        default:         return NSColor(white: 0.32, alpha: 1)
        }
    }

    private static let wireColor = NSColor(calibratedHue: 0.58, saturation: 0.5, brightness: 0.95, alpha: 1)
    private static let portColor = NSColor(calibratedHue: 0.58, saturation: 0.45, brightness: 0.9, alpha: 1)
}

// MARK: - Drawing

extension NodeCanvasView {
    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        ctx.setFillColor(NSColor(white: 0.11, alpha: 1).cgColor)
        ctx.fill(bounds)
        drawGrid(ctx)
        drawConnections(ctx)
        if let dc = draggingConnection { drawDragWire(ctx, dc) }
        drawNodes(ctx)
        if isBoxSelecting { drawBox(ctx) }
        if isCutting { drawCut(ctx) }
    }

    private func drawGrid(_ ctx: CGContext) {
        let step = 50 * zoom
        guard step > 6 else { return }
        let origin = toScreen(0, 0)
        ctx.setStrokeColor(NSColor(white: 0.16, alpha: 1).cgColor)
        ctx.setLineWidth(1)
        var x = origin.x - floor(origin.x / step) * step
        while x < bounds.width { ctx.move(to: CGPoint(x: x, y: 0)); ctx.addLine(to: CGPoint(x: x, y: bounds.height)); x += step }
        var y = origin.y - floor(origin.y / step) * step
        while y < bounds.height { ctx.move(to: CGPoint(x: 0, y: y)); ctx.addLine(to: CGPoint(x: bounds.width, y: y)); y += step }
        ctx.strokePath()
    }

    private func screenCurve(_ s: CGPoint, _ e: CGPoint) -> CGPath {
        let p = CGMutablePath()
        p.move(to: s)
        p.addCurve(to: e, control1: CGPoint(x: s.x + L.cpOffset * zoom, y: s.y),
                   control2: CGPoint(x: e.x - L.cpOffset * zoom, y: e.y))
        return p
    }

    private func drawConnections(_ ctx: CGContext) {
        for c in connections {
            guard let (sw, ew) = connectionEndpoints(c) else { continue }
            ctx.addPath(screenCurve(toScreen(sw.x, sw.y), toScreen(ew.x, ew.y)))
            ctx.setStrokeColor(Self.wireColor.cgColor)
            ctx.setLineWidth(2)
            ctx.strokePath()
        }
    }

    private func drawDragWire(_ ctx: CGContext, _ dc: (node: UUID, port: String, isOutput: Bool)) {
        guard let n = node(dc.node), let pw = portPos(n, dc.port, isInput: !dc.isOutput) else { return }
        let a = toScreen(pw.x, pw.y), b = toScreen(connectionDragPoint.x, connectionDragPoint.y)
        let path = dc.isOutput ? screenCurve(a, b) : screenCurve(b, a)
        ctx.addPath(path)
        ctx.setStrokeColor((hoverDrop != nil ? NSColor.systemGreen : Self.wireColor).cgColor)
        ctx.setLineWidth(2)
        ctx.setLineDash(phase: 0, lengths: [6, 3])
        ctx.strokePath()
        ctx.setLineDash(phase: 0, lengths: [])
    }

    private func drawNodes(_ ctx: CGContext) {
        let showLabels = zoom > 0.45
        for node in nodes {
            guard let pos = positions[node.id] else { continue }
            let tl = toScreen(pos.x, pos.y)
            let w = L.nodeWidth * zoom, h = height(node) * zoom
            let rect = CGRect(x: tl.x, y: tl.y, width: w, height: h)
            let corner = 5 * zoom
            let bodyPath = CGPath(roundedRect: rect, cornerWidth: corner, cornerHeight: corner, transform: nil)

            ctx.addPath(bodyPath)
            ctx.setFillColor(NSColor(white: 0.19, alpha: 1).cgColor)
            ctx.fillPath()

            // header (clipped to rounded body)
            ctx.saveGState()
            ctx.addPath(bodyPath); ctx.clip()
            ctx.setFillColor(headerColor(node).cgColor)
            ctx.fill(CGRect(x: tl.x, y: tl.y, width: w, height: L.headerHeight * zoom))
            ctx.restoreGState()

            // selection / view-node border
            if selected.contains(node.id) {
                ctx.addPath(bodyPath); ctx.setStrokeColor(NSColor.systemYellow.cgColor); ctx.setLineWidth(2); ctx.strokePath()
            } else if viewNodeID == node.id {
                ctx.addPath(bodyPath); ctx.setStrokeColor(NSColor.systemTeal.cgColor); ctx.setLineWidth(2); ctx.strokePath()
            }

            // title
            if showLabels {
                let title = descriptorFor(node.type)?.title ?? node.type
                drawText(title, at: CGPoint(x: tl.x + 6, y: tl.y + 4 * zoom),
                         size: max(9, 11 * zoom), color: .white, weight: .semibold)
            }

            // ports
            let p = ports(node)
            for name in p.inputs {
                guard let wpt = portPos(node, name, isInput: true) else { continue }
                drawPort(ctx, world: wpt, connected: isPortConnected(node.id, name, isOutput: false))
                if showLabels { drawText(name, at: CGPoint(x: toScreen(wpt.x, wpt.y).x + 8, y: toScreen(wpt.x, wpt.y).y - 6 * zoom),
                                         size: max(8, 9 * zoom), color: NSColor(white: 0.7, alpha: 1)) }
            }
            for name in p.outputs {
                guard let wpt = portPos(node, name, isInput: false) else { continue }
                drawPort(ctx, world: wpt, connected: isPortConnected(node.id, name, isOutput: true))
                if showLabels {
                    let s = toScreen(wpt.x, wpt.y)
                    drawText(name, at: CGPoint(x: s.x - 8, y: s.y - 6 * zoom),
                             size: max(8, 9 * zoom), color: NSColor(white: 0.7, alpha: 1), align: .right)
                }
            }
        }
    }

    private func drawPort(_ ctx: CGContext, world: CGPoint, connected: Bool) {
        let s = toScreen(world.x, world.y)
        let r = max(2.5, L.portRadius * zoom)
        let rect = CGRect(x: s.x - r, y: s.y - r, width: 2 * r, height: 2 * r)
        ctx.setFillColor((connected ? Self.portColor : NSColor(white: 0.19, alpha: 1)).cgColor)
        ctx.fillEllipse(in: rect)
        ctx.setStrokeColor(Self.portColor.cgColor)
        ctx.setLineWidth(1.5)
        ctx.strokeEllipse(in: rect)
    }

    private func drawText(_ s: String, at p: CGPoint, size: CGFloat, color: NSColor,
                          align: NSTextAlignment = .left, weight: NSFont.Weight = .regular) {
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: size, weight: weight),
            .foregroundColor: color
        ]
        let str = NSAttributedString(string: s, attributes: attrs)
        if align == .right {
            let sz = str.size()
            str.draw(at: CGPoint(x: p.x - sz.width, y: p.y))
        } else {
            str.draw(at: p)
        }
    }

    private func drawBox(_ ctx: CGContext) {
        let a = toScreen(boxStart.x, boxStart.y), b = toScreen(boxEnd.x, boxEnd.y)
        let r = CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y))
        ctx.setFillColor(NSColor.systemBlue.withAlphaComponent(0.12).cgColor)
        ctx.fill(r)
        ctx.setStrokeColor(NSColor.systemBlue.withAlphaComponent(0.7).cgColor)
        ctx.setLineWidth(1)
        ctx.stroke(r)
    }

    private func drawCut(_ ctx: CGContext) {
        guard cutStroke.count >= 2 else { return }
        ctx.setStrokeColor(NSColor.systemRed.cgColor)
        ctx.setLineWidth(1.5)
        let pts = cutStroke.map { toScreen($0.x, $0.y) }
        ctx.move(to: pts[0])
        for p in pts.dropFirst() { ctx.addLine(to: p) }
        ctx.strokePath()
    }
}

// MARK: - Interaction

extension NodeCanvasView {
    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        let world = toWorld(convert(event.locationInWindow, from: nil))
        let shift = event.modifierFlags.contains(.shift)
        let ctrl = event.modifierFlags.contains(.control)
        didDragNode = false

        if ctrl && !shift { isCutting = true; cutStroke = [world]; needsDisplay = true; return }

        if let hit = portAt(world) {
            draggingConnection = hit; connectionDragPoint = world; needsDisplay = true; return
        }

        if let n = nodeAt(world) {
            if event.clickCount == 2 { onSetViewNode?(n.id); return }
            isDraggingNode = true; draggedNodeID = n.id
            if shift {
                if selected.contains(n.id) { selected.remove(n.id) } else { selected.insert(n.id) }
            } else if !selected.contains(n.id) {
                selected = [n.id]
            }
            onSelect?(selected)
            multiDragOffsets = [:]
            for id in selected { if let pp = positions[id] { multiDragOffsets[id] = CGPoint(x: world.x - pp.x, y: world.y - pp.y) } }
            if let pp = positions[n.id] { dragOffset = CGPoint(x: world.x - pp.x, y: world.y - pp.y) }
        } else if edgeAt(world) != nil {
            if !shift { selected.removeAll(); onSelect?(selected) }
        } else {
            isBoxSelecting = true; boxStart = world; boxEnd = world
            if !shift { selected.removeAll(); onSelect?(selected) }
        }
        lastMouse = event.locationInWindow
        needsDisplay = true
    }

    override func mouseDragged(with event: NSEvent) {
        let world = toWorld(convert(event.locationInWindow, from: nil))
        if isCutting { cutStroke.append(world); needsDisplay = true; return }
        if draggingConnection != nil {
            connectionDragPoint = world
            if let hit = portAt(world), let c = draggingConnection, hit.isOutput != c.isOutput, hit.node != c.node {
                hoverDrop = hit
            } else { hoverDrop = nil }
            needsDisplay = true; return
        }
        if isBoxSelecting { boxEnd = world; needsDisplay = true; return }
        if isDraggingNode {
            didDragNode = true
            if !multiDragOffsets.isEmpty {
                for (id, off) in multiDragOffsets { positions[id] = CGPoint(x: world.x - off.x, y: world.y - off.y) }
            } else if let id = draggedNodeID {
                positions[id] = CGPoint(x: world.x - dragOffset.x, y: world.y - dragOffset.y)
            }
            needsDisplay = true
        }
        lastMouse = event.locationInWindow
    }

    override func mouseUp(with event: NSEvent) {
        if isCutting {
            let hits = edgesHit(byStroke: cutStroke).compactMap { connections.indices.contains($0) ? connections[$0] : nil }
            if !hits.isEmpty { onRun?(EditConnections(remove: hits, add: [])) }
            isCutting = false; cutStroke = []; needsDisplay = true; return
        }
        if let c = draggingConnection {
            let world = toWorld(convert(event.locationInWindow, from: nil))
            if let t = portAt(world), t.isOutput != c.isOutput, t.node != c.node {
                let outNode = c.isOutput ? c.node : t.node
                let outPort = c.isOutput ? c.port : t.port
                let inNode  = c.isOutput ? t.node : c.node
                let inPort  = c.isOutput ? t.port : c.port
                if !Graph(nodes: nodes, connections: connections).wouldCreateCycle(source: outNode, target: inNode) {
                    onRun?(Connect(connection: Connection(fromNode: outNode, fromPort: outPort, toNode: inNode, toPort: inPort)))
                }
            }
            draggingConnection = nil; hoverDrop = nil; needsDisplay = true; return
        }
        if isBoxSelecting {
            let hits = nodesIn(rect: worldRect(boxStart, boxEnd))
            if event.modifierFlags.contains(.shift) { selected.formUnion(hits) } else { selected = Set(hits) }
            onSelect?(selected); isBoxSelecting = false; needsDisplay = true; return
        }
        if isDraggingNode, didDragNode {
            let ids = multiDragOffsets.isEmpty ? [draggedNodeID].compactMap { $0 } : Array(multiDragOffsets.keys)
            for id in ids { if let p = positions[id] { onRun?(MoveNode(nodeID: id, position: p)) } }
        }
        isDraggingNode = false; draggedNodeID = nil; multiDragOffsets = [:]
    }

    override func otherMouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self); isMiddleDragging = true; lastMouse = event.locationInWindow
    }
    override func otherMouseDragged(with event: NSEvent) {
        guard isMiddleDragging else { return }
        panX += event.locationInWindow.x - lastMouse.x
        panY -= event.locationInWindow.y - lastMouse.y
        lastMouse = event.locationInWindow; needsDisplay = true
    }
    override func otherMouseUp(with event: NSEvent) { isMiddleDragging = false }

    override func scrollWheel(with event: NSEvent) {
        let loc = convert(event.locationInWindow, from: nil)
        if event.modifierFlags.contains(.command) || !event.hasPreciseScrollingDeltas {
            let factor: CGFloat = event.scrollingDeltaY > 0 ? 1.1 : (event.scrollingDeltaY < 0 ? 1/1.1 : 1)
            var vp = viewport(); vp.applyZoom(zoom * factor, around: loc, minZoom: Self.minZoom, maxZoom: Self.maxZoom)
            zoom = vp.zoom; panX = vp.panX; panY = vp.panY; needsDisplay = true
        } else {
            panX += event.scrollingDeltaX; panY += event.scrollingDeltaY; needsDisplay = true
        }
    }
    override func magnify(with event: NSEvent) {
        let loc = convert(event.locationInWindow, from: nil)
        var vp = viewport(); vp.applyZoom(zoom * (1 + event.magnification), around: loc, minZoom: Self.minZoom, maxZoom: Self.maxZoom)
        zoom = vp.zoom; panX = vp.panX; panY = vp.panY; needsDisplay = true
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 51 || event.keyCode == 117 { deleteSelection(); return }
        switch event.charactersIgnoringModifiers {
        case "a": selected = Set(nodes.map(\.id)); onSelect?(selected); needsDisplay = true
        case "f": frame(selected.isEmpty ? Set(nodes.map(\.id)) : selected)
        case "h": resetView()
        default: super.keyDown(with: event)
        }
    }

    private func deleteSelection() {
        guard !selected.isEmpty else { return }
        for id in selected { onRun?(DeleteNode(nodeID: id)) }
        selected.removeAll(); onSelect?(selected)
    }

    func resetView() { zoom = 1; panX = 0; panY = 0; frame(Set(nodes.map(\.id))) }

    func frame(_ ids: Set<UUID>) {
        let rects: [CGRect] = ids.compactMap { id in
            guard let p = positions[id], let n = node(id) else { return nil }
            return CGRect(x: p.x, y: p.y, width: L.nodeWidth, height: height(n))
        }
        guard !rects.isEmpty else { return }
        let pad: CGFloat = 60
        let minX = rects.map(\.minX).min()! - pad, maxX = rects.map(\.maxX).max()! + pad
        let minY = rects.map(\.minY).min()! - pad, maxY = rects.map(\.maxY).max()! + pad
        let gw = max(1, maxX - minX), gh = max(1, maxY - minY)
        zoom = max(Self.minZoom, min(Self.maxZoom, min(bounds.width / gw, bounds.height / gh)))
        panX = -((minX + maxX) / 2) * zoom
        panY = -((minY + maxY) / 2) * zoom
        needsDisplay = true
    }

    override func rightMouseDown(with event: NSEvent) {
        let world = toWorld(convert(event.locationInWindow, from: nil))
        pendingAddPos = world
        let menu = NSMenu()

        if let n = nodeAt(world) {
            let del = NSMenuItem(title: "Delete \(descriptorFor(n.type)?.title ?? n.type)",
                                 action: #selector(ctxDeleteNode(_:)), keyEquivalent: "")
            del.target = self; del.representedObject = n.id; menu.addItem(del)
            let view = NSMenuItem(title: "View This Node", action: #selector(ctxViewNode(_:)), keyEquivalent: "")
            view.target = self; view.representedObject = n.id; menu.addItem(view)
            menu.addItem(.separator())
        } else if let idx = edgeAt(world) {
            let del = NSMenuItem(title: "Delete Connection", action: #selector(ctxDeleteEdge(_:)), keyEquivalent: "")
            del.target = self; del.representedObject = idx; menu.addItem(del); menu.addItem(.separator())
        }

        let add = NSMenuItem(title: "Add Node", action: nil, keyEquivalent: "")
        let sub = NSMenu()
        let byCat = Dictionary(grouping: NodeRegistry.all, by: { $0.category })
        for cat in byCat.keys.sorted() {
            let catMenu = NSMenu()
            for d in (byCat[cat] ?? []).sorted(by: { $0.title < $1.title }) {
                let it = NSMenuItem(title: d.title, action: #selector(ctxAddNode(_:)), keyEquivalent: "")
                it.target = self; it.representedObject = d.type; catMenu.addItem(it)
            }
            let catItem = NSMenuItem(title: cat, action: nil, keyEquivalent: "")
            catItem.submenu = catMenu; sub.addItem(catItem)
        }
        add.submenu = sub; menu.addItem(add)
        NSMenu.popUpContextMenu(menu, with: event, for: self)
    }

    @objc private func ctxDeleteNode(_ s: NSMenuItem) {
        if let id = s.representedObject as? UUID { onRun?(DeleteNode(nodeID: id)) }
    }
    @objc private func ctxViewNode(_ s: NSMenuItem) {
        if let id = s.representedObject as? UUID { onSetViewNode?(id) }
    }
    @objc private func ctxDeleteEdge(_ s: NSMenuItem) {
        if let i = s.representedObject as? Int, connections.indices.contains(i) {
            onRun?(EditConnections(remove: [connections[i]], add: []))
        }
    }
    @objc private func ctxAddNode(_ s: NSMenuItem) {
        guard let type = s.representedObject as? String, let d = descriptorFor(type) else { return }
        onRun?(AddNode(node: d.makeNode(at: pendingAddPos)))
    }
}
