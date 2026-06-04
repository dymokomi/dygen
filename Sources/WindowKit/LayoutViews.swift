import SwiftUI

// SwiftUI rendering for the split tree. Ported from dray's LayoutNodeView /
// AreaView / SplitDivider, generalized to WindowKind + an injected content
// builder so WindowKit renders any app's editors without knowing them.

// MARK: - Content Builder (environment slot the app fills in)

public struct WindowContentBuilderKey: EnvironmentKey {
    public static let defaultValue: (WindowKind, UUID) -> AnyView = { kind, _ in
        AnyView(AreaEditorPlaceholder(title: kind.id, systemImage: "square.dashed"))
    }
}

public extension EnvironmentValues {
    var windowContentBuilder: (WindowKind, UUID) -> AnyView {
        get { self[WindowContentBuilderKey.self] }
        set { self[WindowContentBuilderKey.self] = newValue }
    }
}

// MARK: - Dock View (top-level entry point)

/// The whole docking surface: workspace bar + the recursive split tree.
/// The app passes a `content` closure mapping a window kind to its editor view.
public struct DockView: View {
    @ObservedObject public var manager: LayoutManager
    public let content: (WindowKind, UUID) -> AnyView

    public init(manager: LayoutManager, content: @escaping (WindowKind, UUID) -> AnyView) {
        self.manager = manager
        self.content = content
    }

    public var body: some View {
        VStack(spacing: 0) {
            WorkspaceTabBar(manager: manager)
            LayoutNodeView(manager: manager, nodeID: manager.rootNode.id)
        }
        .environment(\.windowContentBuilder, content)
        .background(Theme.bg)
    }
}

// MARK: - Recursive Node View

public struct LayoutNodeView: View {
    @ObservedObject public var manager: LayoutManager
    public let nodeID: UUID

    public init(manager: LayoutManager, nodeID: UUID) {
        self.manager = manager
        self.nodeID = nodeID
    }

    public var body: some View {
        if let node = manager.node(for: nodeID) {
            switch node {
            case .area(let area):  AreaView(manager: manager, area: area)
            case .split(let split): LayoutSplitView(manager: manager, split: split)
            }
        }
    }
}

public struct LayoutSplitView: View {
    @ObservedObject public var manager: LayoutManager
    public let split: SplitState

    public var body: some View {
        GeometryReader { geo in
            let totalSize = split.axis == .horizontal ? geo.size.width : geo.size.height
            let dividerCount = CGFloat(max(split.children.count - 1, 0))
            let available = totalSize - dividerCount * LayoutManager.dividerWidth

            if split.axis == .horizontal {
                HStack(spacing: 0) {
                    ForEach(Array(split.children.enumerated()), id: \.element.id) { index, child in
                        if index > 0 {
                            LayoutDivider(axis: .horizontal, splitID: split.id,
                                          dividerIndex: index - 1, manager: manager, totalSize: available)
                        }
                        LayoutNodeView(manager: manager, nodeID: child.id)
                            .frame(width: max(childSize(index, available), LayoutManager.minimumAreaSize))
                    }
                }
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(split.children.enumerated()), id: \.element.id) { index, child in
                        if index > 0 {
                            LayoutDivider(axis: .vertical, splitID: split.id,
                                          dividerIndex: index - 1, manager: manager, totalSize: available)
                        }
                        LayoutNodeView(manager: manager, nodeID: child.id)
                            .frame(height: max(childSize(index, available), LayoutManager.minimumAreaSize))
                    }
                }
            }
        }
    }

    private func childSize(_ index: Int, _ available: CGFloat) -> CGFloat {
        guard index < split.fractions.count else { return available / CGFloat(split.children.count) }
        return split.fractions[index] * available
    }
}

// MARK: - Area (leaf pane)

public struct AreaView: View {
    @ObservedObject public var manager: LayoutManager
    public let area: AreaState
    @Environment(\.windowContentBuilder) private var contentBuilder

    public var body: some View {
        VStack(spacing: 0) {
            AreaHeader(area: area, manager: manager)
            contentBuilder(area.currentKind, area.id)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(Theme.bg)
    }
}

public struct AreaHeader: View {
    public let area: AreaState
    @ObservedObject public var manager: LayoutManager

    public var body: some View {
        HStack(spacing: 0) {
            // Tabs
            HStack(spacing: 1) {
                ForEach(area.tabs.indices, id: \.self) { i in
                    Button {
                        manager.selectTab(area.id, index: i)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: manager.registry.systemImage(for: area.tabs[i]))
                                .font(.system(size: 12))
                            Text(manager.registry.title(for: area.tabs[i]))
                                .font(.system(size: 12, weight: .medium))
                            if area.tabs.count > 1 || manager.canRemoveArea(area.id) {
                                Image(systemName: "xmark")
                                    .font(.system(size: 7, weight: .bold))
                                    .foregroundColor(Theme.textSecondary.opacity(0.5))
                                    .onTapGesture {
                                        if area.tabs.count > 1 { manager.closeTab(area.id, index: i) }
                                        else { manager.removeArea(area.id) }
                                    }
                            }
                        }
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(i == area.activeTabIndex ? Color.white.opacity(0.08) : Color.clear)
                        .cornerRadius(3)
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(i == area.activeTabIndex ? Theme.textPrimary : Theme.textSecondary)
                }

                // + add tab
                Menu {
                    ForEach(manager.registry.infos) { info in
                        Button {
                            manager.addTab(area.id, kind: info.kind)
                        } label: { Label(info.title, systemImage: info.systemImage) }
                    }
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(Theme.textSecondary)
                        .frame(width: 20, height: 18)
                }
                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
            }

            Spacer()

            // Right-side menu: change editor / split / remove
            Menu {
                Menu("Set Editor") {
                    ForEach(manager.registry.infos) { info in
                        Button {
                            manager.setWindowKind(area.id, to: info.kind)
                        } label: { Label(info.title, systemImage: info.systemImage) }
                    }
                }
                Divider()
                Button {
                    manager.splitArea(area.id, axis: .vertical)
                } label: { Label("Split Horizontal", systemImage: "rectangle.split.1x2") }
                Button {
                    manager.splitArea(area.id, axis: .horizontal)
                } label: { Label("Split Vertical", systemImage: "rectangle.split.2x1") }
                Divider()
                Button(role: .destructive) {
                    manager.removeArea(area.id)
                } label: { Label("Remove Pane", systemImage: "xmark.square") }
                .disabled(!manager.canRemoveArea(area.id))
            } label: {
                Image(systemName: "chevron.down")
                    .font(.system(size: 10))
                    .foregroundColor(Theme.textSecondary)
                    .frame(width: 18, height: 18)
            }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
        }
        .padding(.trailing, 4)
        .frame(height: 24)
        .background(Theme.separator)
        .overlay(alignment: .bottom) { Color(white: 0x10 / 255.0).frame(height: 1) }
    }
}

// MARK: - Divider (proportional drag-resize)

public struct LayoutDivider: View {
    public let axis: Axis
    public let splitID: UUID
    public let dividerIndex: Int
    @ObservedObject public var manager: LayoutManager
    public let totalSize: CGFloat

    @State private var isDragging = false
    @State private var dragStartLocation: CGFloat = 0
    @State private var startFractionA: CGFloat = 0
    @State private var startFractionB: CGFloat = 0

    public var body: some View {
        Rectangle()
            .fill(isDragging ? Theme.textSecondary : Theme.separator)
            .frame(width: axis == .horizontal ? LayoutManager.dividerWidth : nil,
                   height: axis == .vertical ? LayoutManager.dividerWidth : nil)
            .contentShape(Rectangle().inset(by: -(LayoutManager.dividerGrabZone - LayoutManager.dividerWidth) / 2))
            .onHover { hovering in
                if hovering { (axis == .vertical ? NSCursor.resizeUpDown : NSCursor.resizeLeftRight).push() }
                else { NSCursor.pop() }
            }
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { value in
                        if !isDragging {
                            isDragging = true
                            dragStartLocation = axis == .vertical ? value.startLocation.y : value.startLocation.x
                            let fracs = manager.fractions(for: splitID)
                            startFractionA = fracs?[dividerIndex] ?? 0.5
                            startFractionB = fracs?[dividerIndex + 1] ?? 0.5
                        }
                        let current = axis == .vertical ? value.location.y : value.location.x
                        manager.resizeSplit(splitID, dividerIndex: dividerIndex, delta: current - dragStartLocation,
                                            totalSize: totalSize, startA: startFractionA, startB: startFractionB)
                    }
                    .onEnded { _ in
                        isDragging = false
                        dragStartLocation = 0
                        manager.finishResize()
                    }
            )
    }
}

// MARK: - Workspace Tab Bar

public struct WorkspaceTabBar: View {
    @ObservedObject public var manager: LayoutManager
    @State private var showingAddSheet = false
    @State private var newWorkspaceName = ""

    public var body: some View {
        HStack(spacing: 0) {
            ForEach(manager.workspaces) { ws in
                WorkspaceTab(
                    name: ws.name,
                    isActive: ws.name == manager.activeWorkspaceName,
                    onSelect: { manager.switchWorkspace(to: ws.name) },
                    onClose: manager.workspaces.count > 1 ? { manager.deleteWorkspace(name: ws.name) } : nil
                )
            }
            Button {
                newWorkspaceName = "Workspace \(manager.workspaces.count + 1)"
                showingAddSheet = true
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 9))
                    .foregroundColor(Theme.textSecondary)
                    .frame(width: 20, height: 20)
            }
            .buttonStyle(.plain)
            .help("Add workspace")
            .popover(isPresented: $showingAddSheet) {
                VStack(spacing: 8) {
                    Text("New Workspace").font(.system(size: 12, weight: .semibold))
                    TextField("Name", text: $newWorkspaceName)
                        .textFieldStyle(.roundedBorder).frame(width: 150)
                        .onSubmit { addWorkspace() }
                    HStack {
                        Button("Cancel") { showingAddSheet = false }
                        Button("Add") { addWorkspace() }.keyboardShortcut(.defaultAction)
                    }
                }
                .padding(12)
            }
            Spacer()
        }
        .frame(height: 24)
        .background(Theme.separator)
        .overlay(alignment: .bottom) { Color(white: 0x10 / 255.0).frame(height: 1) }
    }

    private func addWorkspace() {
        let name = newWorkspaceName.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { return }
        manager.addWorkspace(name: name)
        showingAddSheet = false
    }
}

private struct WorkspaceTab: View {
    let name: String
    let isActive: Bool
    let onSelect: () -> Void
    let onClose: (() -> Void)?

    var body: some View {
        HStack(spacing: 3) {
            Text(name)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(isActive ? Theme.textPrimary : Theme.textSecondary)
                .lineLimit(1)
            if isActive, let onClose {
                Button(action: onClose) {
                    Image(systemName: "xmark").font(.system(size: 7, weight: .bold))
                        .foregroundColor(Theme.textSecondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 10).padding(.vertical, 4)
        .background(isActive ? Theme.bg : Color.clear)
        .onTapGesture { onSelect() }
    }
}

// MARK: - Placeholder editor

/// "This editor isn't wired up yet" stub, shown by the default content builder.
public struct AreaEditorPlaceholder: View {
    public let title: String
    public let systemImage: String

    public init(title: String, systemImage: String = "square.dashed") {
        self.title = title
        self.systemImage = systemImage
    }

    public var body: some View {
        VStack(spacing: 8) {
            Image(systemName: systemImage)
                .font(.system(size: 24))
                .foregroundColor(Theme.textSecondary.opacity(0.5))
            Text(title)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(Theme.textSecondary.opacity(0.5))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg)
    }
}
