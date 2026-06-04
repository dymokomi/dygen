import SwiftUI
import AppKit

/// Draggable number field with click-to-type editing. Ported from dray's
/// ScrubField: `liveValue` decouples the on-screen number from the bound model
/// during a drag; `onEditingChanged(true/false)` brackets an edit session so the
/// caller can coalesce it into one undo step.
struct ScrubField: View {
    @Binding var value: Double
    var label: String = ""
    var color: Color? = nil
    var speed: Double = 0.01
    var precision: Int = 3
    var onEditingChanged: (Bool) -> Void = { _ in }

    @State private var liveValue: Double?
    @State private var isDragging = false
    @State private var dragStartValue = 0.0
    @State private var isEditing = false
    @State private var editText = ""
    @FocusState private var focused: Bool

    private var displayed: Double { liveValue ?? value }

    var body: some View {
        HStack(spacing: 4) {
            if !label.isEmpty {
                Text(label).font(.system(size: 11)).foregroundColor(color ?? .secondary)
            }
            Group {
                if isEditing {
                    TextField("", text: $editText)
                        .textFieldStyle(.plain)
                        .font(.system(size: 11, design: .monospaced))
                        .focused($focused)
                        .onSubmit { commitEdit() }
                        .onExitCommand { isEditing = false }
                } else {
                    Text(format(displayed))
                        .font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                        .onTapGesture { startEdit() }
                        .gesture(drag)
                }
            }
            .padding(.horizontal, 6).padding(.vertical, 3)
            .background(Color(white: 0.16)).cornerRadius(4)
        }
    }

    private var drag: some Gesture {
        DragGesture(minimumDistance: 2)
            .onChanged { g in
                if !isDragging {
                    isDragging = true
                    dragStartValue = value
                    liveValue = value
                    onEditingChanged(true)
                }
                let fine = NSEvent.modifierFlags.contains(.shift)
                let nv = dragStartValue + Double(g.translation.width) * (fine ? speed * 0.2 : speed)
                liveValue = nv
                value = nv
            }
            .onEnded { _ in
                isDragging = false
                liveValue = nil
                onEditingChanged(false)
            }
    }

    private func format(_ v: Double) -> String {
        String(format: "%.\(max(0, precision))f", v)
    }

    private func startEdit() {
        editText = format(value)
        isEditing = true
        focused = true
    }

    private func commitEdit() {
        if let d = Double(editText) {
            onEditingChanged(true)
            value = d
            onEditingChanged(false)
        }
        isEditing = false
    }
}
