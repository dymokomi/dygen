import SwiftUI

// A Blender-style binary(+) split tree of dockable areas. Ported from dray's
// LayoutManager.swift, generalized from `EditorType` to `WindowKind`.

// MARK: - Split Direction

public enum SplitDirection {
    case left, right, top, bottom
}

// MARK: - Area State (a leaf: one pane with tabbed editors)

public struct AreaState: Identifiable, Codable, Equatable {
    public let id: UUID
    public var tabs: [WindowKind]      // all open tabs
    public var activeTabIndex: Int     // which tab is shown

    public var currentKind: WindowKind {
        get { tabs.indices.contains(activeTabIndex) ? tabs[activeTabIndex] : (tabs.first ?? .placeholder) }
        set {
            if tabs.indices.contains(activeTabIndex) {
                tabs[activeTabIndex] = newValue
            }
        }
    }

    public init(id: UUID = UUID(), kind: WindowKind) {
        self.id = id
        self.tabs = [kind]
        self.activeTabIndex = 0
    }

    public mutating func addTab(_ kind: WindowKind) {
        tabs.append(kind)
        activeTabIndex = tabs.count - 1
    }

    public mutating func closeTab(at index: Int) {
        guard tabs.count > 1, tabs.indices.contains(index) else { return }
        tabs.remove(at: index)
        if activeTabIndex >= tabs.count {
            activeTabIndex = tabs.count - 1
        }
    }
}

// MARK: - Split State (an internal node: children + axis + proportions)

public struct SplitState: Identifiable, Equatable {
    public let id: UUID
    public var axis: Axis
    public var children: [LayoutNode]
    public var fractions: [CGFloat]

    public init(id: UUID = UUID(), axis: Axis, children: [LayoutNode], fractions: [CGFloat]? = nil) {
        self.id = id
        self.axis = axis
        self.children = children
        let count = children.count
        self.fractions = fractions ?? Array(repeating: 1.0 / CGFloat(count), count: count)
    }
}

extension SplitState: Codable {
    private enum CodingKeys: String, CodingKey { case id, axis, children, fractions }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        // SwiftUI.Axis isn't Codable, so persist it as a string.
        let axisStr = try c.decode(String.self, forKey: .axis)
        axis = axisStr == "horizontal" ? .horizontal : .vertical
        children = try c.decode([LayoutNode].self, forKey: .children)
        fractions = try c.decode([CGFloat].self, forKey: .fractions)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(axis == .horizontal ? "horizontal" : "vertical", forKey: .axis)
        try c.encode(children, forKey: .children)
        try c.encode(fractions, forKey: .fractions)
    }
}

// MARK: - Layout Node (area | split)

public indirect enum LayoutNode: Identifiable, Codable, Equatable {
    case area(AreaState)
    case split(SplitState)

    public var id: UUID {
        switch self {
        case .area(let s): return s.id
        case .split(let s): return s.id
        }
    }

    private enum CodingKeys: String, CodingKey { case type, area, split }
    private enum NodeType: String, Codable { case area, split }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(NodeType.self, forKey: .type) {
        case .area:  self = .area(try c.decode(AreaState.self, forKey: .area))
        case .split: self = .split(try c.decode(SplitState.self, forKey: .split))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .area(let s):
            try c.encode(NodeType.area, forKey: .type)
            try c.encode(s, forKey: .area)
        case .split(let s):
            try c.encode(NodeType.split, forKey: .type)
            try c.encode(s, forKey: .split)
        }
    }
}

// MARK: - Workspace & persisted file

public struct Workspace: Codable, Identifiable, Equatable {
    public var id: String { name }
    public var name: String
    public var rootNode: LayoutNode

    public init(name: String, rootNode: LayoutNode) {
        self.name = name
        self.rootNode = rootNode
    }
}

public struct LayoutFile: Codable {
    public static let currentVersion = 1
    public var version: Int = LayoutFile.currentVersion
    public var workspaces: [Workspace]
    public var activeWorkspaceName: String
}
