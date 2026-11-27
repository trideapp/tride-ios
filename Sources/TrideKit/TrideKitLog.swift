/// How much TrideKit writes to the log, and where: one level and one handler per process.
///
/// ## Public API
/// - `TrideKitLog.level` (`TrideKitLogLevel`: `off`, `error`, `warning`, `info`, `debug`; default
///   `warning`) — set from any thread at any time. The launch argument `-TrideKitLogLevel <level>`
///   (an Xcode scheme's argument: this launch only) wins over the code while it is given.
/// - `TrideKitLog.handler` — `@Sendable (TrideKitLogEntry) -> Void` gets every line that passes the
///   level instead of the system log; `nil` = the system log (subsystem `studio.tride.kit`, the
///   line's category).
///
/// ## Non-obvious constraints
/// - The level and the writer live in the bridge's lowest layer (`TrideLog.mm`): the ObjC++ bridge
///   obeys them without this file; here the handler is stored (`OSAllocatedUnfairLock`) and
///   installed as the bridge's sink. Where the bridge is not linked (the shipped package on macOS)
///   the same level, launch argument and handler live here, with `os.Logger` as the writer.
/// - The handler runs on the writing thread (main, the preparation queue), never under the core's
///   lock; it must be quick and must not call the scene. It is called outside the storage's lock.
/// - The scene's tick writes no line at any level. The core's notices are written as `warning`,
///   category `core`, once each, and reach `subscribeDiagnostics` whatever the level
///   (`CoreNotices.swift`).
/// - `TrideLog` (internal) is the only way TrideKit's Swift code writes: `@autoclosure` text, built
///   only when the level passes.

import Foundation
import os

#if canImport(TrideKitFilamentBridge)
import TrideKitFilamentBridge
#else
// Without the bridge (the shipped package on macOS, where TrideKit is a re-export of TrideCore):
// the bridge's level, launch argument and writer, here; the handler is the same storage.
typealias TrideLogLevel = TrideKitLogLevel
private let codeLevel = OSAllocatedUnfairLock<TrideKitLogLevel?>(initialState: nil)
/// The launch argument, read once (this launch only); it wins over the code while given.
private let argumentLevel: TrideKitLogLevel? = (UserDefaults.standard
    .volatileDomain(forName: UserDefaults.argumentDomain)["TrideKitLogLevel"] as? String)
    .flatMap(TrideKitLogLevel.init(name:))
private let writtenOnce = OSAllocatedUnfairLock<Set<String>>(initialState: [])

func TrideLogLevelNamed(_ name: String) -> Int {
    ["off": 0, "error": 1, "warning": 2, "info": 3, "debug": 4][name.lowercased()] ?? -1
}

func TrideLogCurrentLevel() -> TrideKitLogLevel { argumentLevel ?? codeLevel.withLock { $0 } ?? .warning }

func TrideLogSetLevel(_ level: TrideKitLogLevel) { codeLevel.withLock { $0 = level } }

func TrideLogEnabled(_ level: TrideKitLogLevel) -> Bool { level > .off && level <= TrideLogCurrentLevel() }

func TrideLogWriteOnce(_ level: TrideKitLogLevel, _ category: String, _ message: String) {
    guard TrideLogEnabled(level),
          writtenOnce.withLock({ $0.insert(category + "\n" + message).inserted }) else { return }
    TrideLogWrite(level, category, message)
}

/// The handler, else the system log as the bridge writes it (warning and error as `.error`).
func TrideLogWrite(_ level: TrideKitLogLevel, _ category: String, _ message: String) {
    guard TrideLogEnabled(level) else { return }
    if let handler = handlerStorage.withLock({ $0 }) {
        handler(TrideKitLogEntry(level: level, category: category, message: message))
        return
    }
    let type: OSLogType = level <= .warning ? .error : (level == .info ? .info : .debug)
    Logger(subsystem: "studio.tride.kit", category: category).log(level: type, "\(message, privacy: .public)")
}
#endif

/// How much TrideKit writes: each level includes the ones before it.
public enum TrideKitLogLevel: Int, Sendable, CaseIterable, Comparable, CustomStringConvertible {
    /// Nothing.
    case off = 0
    /// What was asked could not be done: a scene that did not load, no Metal.
    case error = 1
    /// Degraded output, the scenes' diagnostics, the core's notices. The default.
    case warning = 2
    /// A line per scene event: loaded, engine made, camera switched.
    case info = 3
    /// Details of a load: lights, module meshes, render settings.
    case debug = 4

    /// `off` / `error` / `warning` / `info` / `debug`, any case — as the launch argument takes it.
    public init?(name: String) {
        guard let level = TrideKitLogLevel(rawValue: TrideLogLevelNamed(name)) else { return nil }
        self = level
    }

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }

    public var description: String {
        switch self {
        case .off: "off"
        case .error: "error"
        case .warning: "warning"
        case .info: "info"
        case .debug: "debug"
        }
    }
}

/// One line TrideKit writes: its level, its category (`TrideKitScene`, `PointerDispatcher`,
/// `FilamentBridge`, `core`, …) and its text.
public struct TrideKitLogEntry: Sendable, Equatable {
    public let level: TrideKitLogLevel
    public let category: String
    public let message: String

    public init(level: TrideKitLogLevel, category: String, message: String) {
        self.level = level
        self.category = category
        self.message = message
    }
}

/// TrideKit's log settings for the process.
public enum TrideKitLog {
    /// How much TrideKit writes: the launch argument `-TrideKitLogLevel <level>` while it is given,
    /// else what the code set, else `warning`. Any thread.
    public static var level: TrideKitLogLevel {
        get { TrideKitLogLevel(rawValue: TrideLogCurrentLevel().rawValue) ?? .warning }
        set { TrideLogSetLevel(TrideLogLevel(rawValue: newValue.rawValue) ?? .warning) }
    }

    /// Where the lines that pass `level` go; `nil` = the system log. Called on the writing thread:
    /// keep it quick and do not call the scene from it. Any thread.
    public static var handler: (@Sendable (TrideKitLogEntry) -> Void)? {
        get { handlerStorage.withLock { $0 } }
        set {
            #if canImport(TrideKitFilamentBridge)
            // The handler and the bridge's sink change together: two setters at once leave both as
            // the last one set, never the sink installed over no handler (which drops every line).
            handlerStorage.withLock {
                $0 = newValue
                TrideLogSetSink(newValue == nil ? nil : handlerSink)
            }
            #else
            handlerStorage.withLock { $0 = newValue }
            #endif
        }
    }
}

private let handlerStorage = OSAllocatedUnfairLock<(@Sendable (TrideKitLogEntry) -> Void)?>(initialState: nil)

#if canImport(TrideKitFilamentBridge)
/// The bridge's sink while a handler is set: the entry, then the handler, outside the lock.
private let handlerSink: TrideLogSink = { level, category, message in
    guard let handler = handlerStorage.withLock({ $0 }) else { return }
    handler(TrideKitLogEntry(
        level: TrideKitLogLevel(rawValue: level.rawValue) ?? .debug,
        category: String(cString: category),
        message: String(cString: message)))
}
#endif

/// TrideKit's Swift lines, through the bridge's writer: the text is built only when the level passes.
enum TrideLog {
    static func error(_ category: String, _ message: @autoclosure () -> String) {
        write(.error, category, message)
    }

    static func warning(_ category: String, _ message: @autoclosure () -> String) {
        write(.warning, category, message)
    }

    static func info(_ category: String, _ message: @autoclosure () -> String) {
        write(.info, category, message)
    }

    static func debug(_ category: String, _ message: @autoclosure () -> String) {
        write(.debug, category, message)
    }

    /// A warning once per process for each distinct text: a line on a path a tick may take again.
    static func warningOnce(_ category: String, _ message: @autoclosure () -> String) {
        guard TrideLogEnabled(.warning) else { return }
        TrideLogWriteOnce(.warning, category, message())
    }

    @inline(__always)
    private static func write(_ level: TrideLogLevel, _ category: String, _ message: () -> String) {
        guard TrideLogEnabled(level) else { return }
        TrideLogWrite(level, category, message())
    }
}
