import XCTest
@testable import WindowKit

final class LayoutTests: XCTestCase {

    private func registry() -> WindowKindRegistry {
        WindowKindRegistry([
            WindowKindInfo(WindowKind("a"), title: "A", systemImage: "a"),
            WindowKindInfo(WindowKind("b"), title: "B", systemImage: "b"),
        ])
    }

    private func manager(_ url: URL) -> LayoutManager {
        let ws = Workspace(name: "Default", rootNode: .area(AreaState(kind: WindowKind("a"))))
        return LayoutManager(registry: registry(), storeURL: url, defaultWorkspace: ws)
    }

    private func tmpURL() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("wk-\(UUID().uuidString).json")
    }

    // MARK: - Tabs

    func testTabsAddAndClose() {
        var area = AreaState(kind: WindowKind("a"))
        area.addTab(WindowKind("b"))
        XCTAssertEqual(area.tabs.count, 2)
        XCTAssertEqual(area.activeTabIndex, 1)
        XCTAssertEqual(area.currentKind, WindowKind("b"))

        area.closeTab(at: 1)
        XCTAssertEqual(area.tabs.count, 1)
        // The last remaining tab can't be closed.
        area.closeTab(at: 0)
        XCTAssertEqual(area.tabs.count, 1)
    }

    // MARK: - Splitting / removal

    func testSplitCreatesTwoAreas() {
        let m = manager(tmpURL())
        let rootID = m.rootNode.id
        m.splitArea(rootID, direction: .right)
        guard case .split(let s) = m.rootNode else { return XCTFail("expected a split node") }
        XCTAssertEqual(s.children.count, 2)
        XCTAssertEqual(s.axis, .horizontal)
        XCTAssertEqual(s.fractions, [0.5, 0.5])
    }

    func testRemoveAreaCollapsesSplit() {
        let m = manager(tmpURL())
        m.splitArea(m.rootNode.id, direction: .right)
        guard case .split(let s) = m.rootNode else { return XCTFail("expected split") }
        m.removeArea(s.children[1].id)
        guard case .area = m.rootNode else { return XCTFail("removing one child should collapse to a single area") }
    }

    func testCannotRemoveOnlyArea() {
        let m = manager(tmpURL())
        XCTAssertFalse(m.canRemoveArea(m.rootNode.id))
    }

    // MARK: - Resizing

    func testResizePreservesSumAndMovesDivider() {
        let m = manager(tmpURL())
        m.splitArea(m.rootNode.id, direction: .right)
        guard case .split(let s) = m.rootNode else { return XCTFail() }
        m.resizeSplit(s.id, dividerIndex: 0, delta: 100, totalSize: 1000, startA: 0.5, startB: 0.5)
        let f = try! XCTUnwrap(m.fractions(for: s.id))
        XCTAssertEqual(f[0] + f[1], 1.0, accuracy: 0.0001)
        XCTAssertGreaterThan(f[0], 0.5) // dragging right grows the left pane
    }

    func testResizeKeepsPanesPositiveAtExtremes() {
        // An extreme drag must never produce a zero/negative fraction or change
        // the total. (The hard 100pt visual minimum is enforced separately by
        // the split view's frame, not by the fraction math.)
        let m = manager(tmpURL())
        m.splitArea(m.rootNode.id, direction: .right)
        guard case .split(let s) = m.rootNode else { return XCTFail() }
        m.resizeSplit(s.id, dividerIndex: 0, delta: -10_000, totalSize: 1000, startA: 0.5, startB: 0.5)
        let f = try! XCTUnwrap(m.fractions(for: s.id))
        XCTAssertGreaterThan(f[0], 0)
        XCTAssertGreaterThan(f[1], 0)
        XCTAssertLessThan(f[0], f[1]) // dragging far left shrank the left pane
        XCTAssertEqual(f[0] + f[1], 1.0, accuracy: 0.0001)
    }

    // MARK: - Persistence

    func testSaveRestoreRoundTrip() {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let m = manager(url)
        m.splitArea(m.rootNode.id, direction: .bottom) // vertical split
        m.saveLayout()
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

        let restored = manager(url)
        restored.restoreLayout()
        XCTAssertEqual(restored.rootNode, m.rootNode)
        guard case .split(let s) = restored.rootNode else { return XCTFail("restored layout should be a split") }
        XCTAssertEqual(s.axis, .vertical)
    }

    func testRestoreWithNoFileKeepsDefault() {
        let m = manager(tmpURL()) // file doesn't exist
        let before = m.rootNode
        m.restoreLayout()
        XCTAssertEqual(m.rootNode, before) // unchanged default
    }

    func testWorkspacesPersist() {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let m = manager(url)
        m.addWorkspace(name: "Second")
        m.saveLayout()

        let restored = manager(url)
        restored.restoreLayout()
        XCTAssertEqual(restored.workspaces.count, 2)
        XCTAssertEqual(restored.activeWorkspaceName, "Second")
    }
}
