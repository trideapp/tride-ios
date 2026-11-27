/// The core's degrade notices in TrideKit: pulled from the runtime, written once
/// to the log and handed to every scene's `subscribeDiagnostics`.
///
/// ## Internal API
/// - `CoreNotices.pull()` — take the runtime's new notices (`takeCoreNotices`), keep them in the
///   process list and write each to the log (`warning`, category `core`); any thread.
/// - `CoreNotices.since(_:)` — the process list from a scene's cursor on.
/// - `CoreNotices.diagnostic(_:)` — a notice as a scene diagnostic (`.runtime`, code `core.notice`).
/// - `CoreNotices.open(_:)` — a scene alive now (held weakly): new notices reach it on the main thread.
///
/// ## Non-obvious constraints
/// - Pull, never push: the runtime queues notices (it may raise one under a scene's lock) and the
///   bridge takes them after each tick and load, outside any runtime call. The first pull happens
///   before the first scene loads, so from then on the runtime writes no notice to the log itself.
/// - The core raises each notice once per process: one missed by the log at `off` is not written
///   again later; the diagnostics get every one, and a scene opened later gets the list so far.

import Foundation
import os

#if canImport(TrideCore)
import TrideCore
#endif

// The scenes it hands notices to exist only with the bridge (not in the shipped package on macOS).
#if canImport(TrideKitFilamentBridge)

enum CoreNotices {
    static let diagnosticCode = "core.notice"

    /// A scene alive now, held weakly.
    private final class WeakScene: @unchecked Sendable {
        weak var scene: TrideKitScene?
        init(_ scene: TrideKitScene) { self.scene = scene }
    }

    private struct State: @unchecked Sendable {
        var all: [String] = []
        var scenes: [WeakScene] = []
    }

    private static let state = OSAllocatedUnfairLock(initialState: State())

    /// The runtime's notices since the last pull, now in the process list and the log. When there
    /// are new ones, every open scene hands them to its diagnostics on the main thread soon after
    /// (the scene that pulled does so at once, in `deliverDiagnostics`).
    @discardableResult
    static func pull() -> [String] {
        let fresh: [String] = state.withLock { state in
            let taken = takeCoreNotices()
            state.all.append(contentsOf: taken)
            return taken
        }
        guard !fresh.isEmpty else { return fresh }
        for message in fresh { TrideLog.warning("core", message) }
        DispatchQueue.main.async {
            for scene in openScenes() { scene.deliverDiagnostics() }
        }
        return fresh
    }

    /// The process's notices from index `cursor` on, and the cursor after them.
    static func since(_ cursor: Int) -> (notices: [String], next: Int) {
        state.withLock { state in
            let from = min(max(cursor, 0), state.all.count)
            return (Array(state.all[from...]), state.all.count)
        }
    }

    static func diagnostic(_ message: String) -> TrideKitDiagnostic {
        TrideKitDiagnostic(kind: .runtime, subject: "core", message: message, code: diagnosticCode)
    }

    /// `scene` is open: it gets the notices pulled from now on (and the list so far, by its cursor).
    static func open(_ scene: TrideKitScene) {
        state.withLock { state in
            state.scenes.removeAll { $0.scene == nil }
            state.scenes.append(WeakScene(scene))
        }
    }

    private static func openScenes() -> [TrideKitScene] {
        state.withLock { $0.scenes.compactMap(\.scene) }
    }
}

#endif
