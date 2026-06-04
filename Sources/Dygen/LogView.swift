import SwiftUI
import AppKit
import DygenCore

/// The Log window. Snapshots `AppLog.shared.entries` on every `appLogChanged`
/// notification, auto-scrolls to the newest line. Ported from dray's LogView.
struct LogView: View {
    @State private var entries: [AppLog.Entry] = AppLog.shared.entries

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Text("\(entries.count) lines")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Copy") { copyAll() }.buttonStyle(.plain).font(.system(size: 10))
                Button("Clear") { AppLog.shared.clear() }.buttonStyle(.plain).font(.system(size: 10))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(white: 0.16))

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(entries) { e in
                            HStack(alignment: .top, spacing: 6) {
                                Text(timeString(e.timestamp))
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundStyle(.tertiary)
                                Text(e.level.rawValue)
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundStyle(color(for: e.level))
                                    .frame(width: 36, alignment: .leading)
                                Text(e.message)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(.primary)
                                    .textSelection(.enabled)
                                Spacer(minLength: 0)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 1)
                            .id(e.id)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .onChange(of: entries.count) {
                    if let last = entries.last { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
        .background(Color(white: 0.12))
        .onReceive(NotificationCenter.default.publisher(for: .appLogChanged)) { _ in
            entries = AppLog.shared.entries
        }
    }

    private func color(for level: AppLog.Entry.Level) -> Color {
        switch level {
        case .info:    return .secondary
        case .warning: return .yellow
        case .error:   return .red
        case .command: return .cyan
        }
    }

    private func timeString(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: date)
    }

    private func copyAll() {
        let text = entries.map { "[\($0.level.rawValue)] \($0.message)" }.joined(separator: "\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
