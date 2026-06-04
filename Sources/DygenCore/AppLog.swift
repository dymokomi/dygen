import Foundation

/// Posted after every log append (and on `clear()`). `LogView` refreshes its
/// snapshot on this. Ported/simplified from dray's AppLog.
public extension Notification.Name {
    static let appLogChanged = Notification.Name("dygen.AppLogChanged")
}

/// Process-wide logging facade: an in-memory ring (for the Log window), stderr,
/// and a truncate-on-launch file at `~/Library/Application Support/Dygen/dygen.log`.
public final class AppLog: @unchecked Sendable {
    public static let shared = AppLog()

    public struct Entry: Identifiable, Sendable {
        public let id = UUID()
        public let timestamp: Date
        public let level: Level
        public let message: String

        public enum Level: String, Sendable {
            case info = "INFO"
            case warning = "WARN"
            case error = "ERROR"
            case command = "CMD"
        }
    }

    private static let limit = 2000
    private let lock = NSLock()
    private var ring: [Entry] = []
    private let fileURL: URL?
    private let fileQueue = DispatchQueue(label: "dygen.applog.file")

    /// Thread-safe snapshot of the current ring.
    public var entries: [Entry] {
        lock.lock(); defer { lock.unlock() }
        return ring
    }

    private init() {
        fileURL = Self.setupFile()
    }

    public func log(_ message: String, level: Entry.Level = .info) {
        let entry = Entry(timestamp: Date(), level: level, message: message)
        lock.lock()
        ring.append(entry)
        if ring.count > Self.limit { ring.removeFirst(ring.count - Self.limit) }
        lock.unlock()
        NotificationCenter.default.post(name: .appLogChanged, object: nil)
        FileHandle.standardError.write(Data("[\(level.rawValue)] \(message)\n".utf8))
        if let fileURL { fileQueue.async { Self.appendFile(fileURL, entry) } }
    }

    public func warn(_ m: String) { log(m, level: .warning) }
    public func error(_ m: String) { log(m, level: .error) }
    public func command(_ m: String) { log(m, level: .command) }

    public func clear() {
        lock.lock(); ring.removeAll(); lock.unlock()
        NotificationCenter.default.post(name: .appLogChanged, object: nil)
    }

    // MARK: - File sink (truncate on launch; simple per-write open/close)

    private static func setupFile() -> URL? {
        guard let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        else { return nil }
        let dir = base.appendingPathComponent("Dygen", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("dygen.log")
        FileManager.default.createFile(atPath: url.path, contents: nil) // truncate
        return url
    }

    private static func appendFile(_ url: URL, _ e: Entry) {
        guard let h = try? FileHandle(forWritingTo: url) else { return }
        defer { try? h.close() }
        h.seekToEndOfFile()
        try? h.write(contentsOf: Data("[\(e.level.rawValue)] \(e.message)\n".utf8))
    }
}
