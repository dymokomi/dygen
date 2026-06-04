import SwiftUI

/// Owns the live split tree, mutates it, and persists/restores it to disk.
/// Ported from dray's LayoutManager; storage path and default layout are
/// injected so WindowKit carries no app-specific knowledge.
public final class LayoutManager: ObservableObject {
    @Published public var rootNode: LayoutNode
    @Published public var workspaces: [Workspace]
    @Published public var activeWorkspaceName: String

    /// Resolves window-kind display metadata for the chrome (tabs, menus).
    public let registry: WindowKindRegistry

    /// Full file URL the layout JSON is written to / read from.
    public let storeURL: URL

    public static let minimumAreaSize: CGFloat = 100
    public static let dividerWidth: CGFloat = 4
    public static let dividerGrabZone: CGFloat = 8

    public init(registry: WindowKindRegistry, storeURL: URL, defaultWorkspace: Workspace) {
        self.registry = registry
        self.storeURL = storeURL
        self.rootNode = defaultWorkspace.rootNode
        self.workspaces = [defaultWorkspace]
        self.activeWorkspaceName = defaultWorkspace.name
    }

    // MARK: - Node Lookup

    public func node(for id: UUID) -> LayoutNode? { findNode(in: rootNode, id: id) }

    private func findNode(in node: LayoutNode, id: UUID) -> LayoutNode? {
        if node.id == id { return node }
        if case .split(let state) = node {
            for child in state.children {
                if let found = findNode(in: child, id: id) { return found }
            }
        }
        return nil
    }

    // MARK: - Editor Kind / Tabs

    public func setWindowKind(_ areaID: UUID, to newKind: WindowKind) {
        rootNode = mutateArea(in: rootNode, areaID: areaID) { area in
            var u = area; u.currentKind = newKind; return u
        }
        scheduleSave()
    }

    public func addTab(_ areaID: UUID, kind: WindowKind) {
        rootNode = mutateArea(in: rootNode, areaID: areaID) { area in
            var u = area; u.addTab(kind); return u
        }
        scheduleSave()
    }

    public func selectTab(_ areaID: UUID, index: Int) {
        rootNode = mutateArea(in: rootNode, areaID: areaID) { area in
            var u = area; u.activeTabIndex = index; return u
        }
        scheduleSave()
    }

    public func closeTab(_ areaID: UUID, index: Int) {
        rootNode = mutateArea(in: rootNode, areaID: areaID) { area in
            var u = area; u.closeTab(at: index); return u
        }
        scheduleSave()
    }

    // MARK: - Splitting / Removal

    public func splitArea(_ areaID: UUID, direction: SplitDirection) {
        guard let node = findNode(in: rootNode, id: areaID),
              case .area(let area) = node else { return }

        let newArea = AreaState(kind: area.currentKind)
        let axis: Axis = (direction == .left || direction == .right) ? .horizontal : .vertical
        let children: [LayoutNode] = (direction == .left || direction == .top)
            ? [.area(newArea), .area(area)]
            : [.area(area), .area(newArea)]
        let splitNode = LayoutNode.split(SplitState(axis: axis, children: children, fractions: [0.5, 0.5]))

        rootNode = replaceNode(in: rootNode, targetID: areaID, with: splitNode)
        scheduleSave()
    }

    /// Convenience: split along an axis (used by the header menu).
    public func splitArea(_ areaID: UUID, axis: Axis) {
        splitArea(areaID, direction: axis == .horizontal ? .right : .bottom)
    }

    public func removeArea(_ areaID: UUID) {
        guard canRemoveArea(areaID) else { return }
        rootNode = removeNode(from: rootNode, targetID: areaID)
        scheduleSave()
    }

    public func canRemoveArea(_ areaID: UUID) -> Bool {
        // Can't remove the only area (root is a single leaf).
        if case .area = rootNode { return false }
        return true
    }

    // MARK: - Edge Resizing

    public func fractions(for splitID: UUID) -> [CGFloat]? {
        guard let node = findNode(in: rootNode, id: splitID),
              case .split(let state) = node else { return nil }
        return state.fractions
    }

    public func resizeSplit(_ splitID: UUID, dividerIndex: Int, delta: CGFloat, totalSize: CGFloat, startA: CGFloat, startB: CGFloat) {
        rootNode = mutateSplit(in: rootNode, splitID: splitID) { split in
            var u = split
            guard dividerIndex >= 0, dividerIndex < u.fractions.count - 1 else { return u }

            let minFraction = Self.minimumAreaSize / max(totalSize, 1)
            let combined = startA + startB
            let deltaFraction = delta / max(totalSize, 1)

            var newA = max(startA + deltaFraction, minFraction)
            var newB = max(startB - deltaFraction, minFraction)
            let scale = combined / (newA + newB)
            newA *= scale
            newB *= scale

            u.fractions[dividerIndex] = newA
            u.fractions[dividerIndex + 1] = newB
            return u
        }
    }

    public func finishResize() { scheduleSave() }

    // MARK: - Workspaces

    public func switchWorkspace(to name: String) {
        saveCurrentWorkspace()
        guard let ws = workspaces.first(where: { $0.name == name }) else { return }
        rootNode = ws.rootNode
        activeWorkspaceName = name
        scheduleSave()
    }

    public func addWorkspace(name: String) {
        saveCurrentWorkspace()
        workspaces.append(Workspace(name: name, rootNode: rootNode))
        activeWorkspaceName = name
        scheduleSave()
    }

    public func deleteWorkspace(name: String) {
        guard workspaces.count > 1, name != activeWorkspaceName else { return }
        workspaces.removeAll { $0.name == name }
        scheduleSave()
    }

    private func saveCurrentWorkspace() {
        if let idx = workspaces.firstIndex(where: { $0.name == activeWorkspaceName }) {
            workspaces[idx].rootNode = rootNode
        }
    }

    // MARK: - Persistence

    public func saveLayout() {
        saveCurrentWorkspace()
        let file = LayoutFile(workspaces: workspaces, activeWorkspaceName: activeWorkspaceName)
        guard let data = try? JSONEncoder().encode(file) else { return }
        let formatted = (try? JSONSerialization.jsonObject(with: data))
            .flatMap { try? JSONSerialization.data(withJSONObject: $0, options: [.prettyPrinted, .sortedKeys]) }
        try? FileManager.default.createDirectory(at: storeURL.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? (formatted ?? data).write(to: storeURL)
    }

    public func restoreLayout() {
        guard let data = try? Data(contentsOf: storeURL),
              let file = try? JSONDecoder().decode(LayoutFile.self, from: data),
              file.version == LayoutFile.currentVersion,
              !file.workspaces.isEmpty else { return }
        workspaces = file.workspaces
        activeWorkspaceName = file.activeWorkspaceName
        if let ws = workspaces.first(where: { $0.name == activeWorkspaceName }) {
            rootNode = ws.rootNode
        } else if let first = workspaces.first {
            rootNode = first.rootNode
            activeWorkspaceName = first.name
        }
    }

    private var saveWorkItem: DispatchWorkItem?

    private func scheduleSave() {
        saveWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in self?.saveLayout() }
        saveWorkItem = item
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0, execute: item)
    }

    // MARK: - Tree Mutation Helpers

    private func mutateArea(in node: LayoutNode, areaID: UUID, transform: (AreaState) -> AreaState) -> LayoutNode {
        switch node {
        case .area(let s):
            return s.id == areaID ? .area(transform(s)) : node
        case .split(var s):
            s.children = s.children.map { mutateArea(in: $0, areaID: areaID, transform: transform) }
            return .split(s)
        }
    }

    private func mutateSplit(in node: LayoutNode, splitID: UUID, transform: (SplitState) -> SplitState) -> LayoutNode {
        switch node {
        case .area:
            return node
        case .split(var s):
            if s.id == splitID { return .split(transform(s)) }
            s.children = s.children.map { mutateSplit(in: $0, splitID: splitID, transform: transform) }
            return .split(s)
        }
    }

    private func replaceNode(in node: LayoutNode, targetID: UUID, with replacement: LayoutNode) -> LayoutNode {
        if node.id == targetID { return replacement }
        switch node {
        case .area:
            return node
        case .split(var s):
            s.children = s.children.map { replaceNode(in: $0, targetID: targetID, with: replacement) }
            return .split(s)
        }
    }

    private func removeNode(from node: LayoutNode, targetID: UUID) -> LayoutNode {
        switch node {
        case .area:
            return node
        case .split(var s):
            if let idx = s.children.firstIndex(where: { $0.id == targetID }) {
                s.children.remove(at: idx)
                s.fractions.remove(at: idx)
                let total = s.fractions.reduce(0, +)
                if total > 0 { s.fractions = s.fractions.map { $0 / total } }
                if s.children.count == 1 { return s.children[0] }
                return .split(s)
            }
            s.children = s.children.map { removeNode(from: $0, targetID: targetID) }
            if s.children.count == 1 { return s.children[0] }
            return .split(s)
        }
    }
}
