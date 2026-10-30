/// TrideKit's thread rule, checked when the app runs (the same rule as TrideKit for Android).
///
/// ## Public API
/// - `TrideKitThreadRule.refusal` (`@_spi(TrideTesting)`) — how a broken rule ends; a test swaps it
///   to watch the refusal without losing its process.
///
/// ## Non-obvious constraints
/// - Calls that step, draw, touch, subscribe or attach (`play`, `tickOnce`, `advance`, `reset`,
///   `pointer`, `subscribeEvents`, audio, pictures, …) belong to the main thread: the frame runs
///   there and the scene's dispatch state, its handlers and Filament take no lock. They are
///   `@MainActor`, so a Swift 6 app cannot call them elsewhere; a Swift 5 app can (a delegate or a
///   completion handler that runs on a background queue), and such a call stops the app AT the call
///   with a message naming it, instead of a crash in a later frame far from the cause.
/// - The runtime-only calls (`setBinding`, `trigger`, `setInputs`, `getBinding`, `currentState`, …)
///   are `nonisolated`: they reach the core through its own lock, so any thread may make them.
/// - Checked at the public entry points only, never in the per-frame path.

import Foundation

/// The check behind every main-thread call of `TrideKitScene`.
@_spi(TrideTesting) public enum TrideKitThreadRule {
    /// Stops the process with the message (`fatalError`: kept in Release builds and in crash reports).
    /// Tests swap it; nothing else writes it.
    public nonisolated(unsafe) static var refusal: (String) -> Never = { message in fatalError(message) }

    /// `call` (the entry point's `#function`) runs on the main thread, or the rule refuses it.
    @inline(__always)
    static func requireMain(_ call: StaticString = #function) {
        guard !Thread.isMainThread else { return }
        refusal(message(call))
    }

    static func message(_ call: StaticString) -> String {
        let queue = String(cString: __dispatch_queue_get_label(nil))
        let thread = Thread.current.name.flatMap { $0.isEmpty ? nil : $0 } ?? queue
        return "TrideKitScene.\(call) ran off the main thread (on '\(thread)'): calls that step, draw, touch, "
            + "subscribe or attach belong to the main thread - call it from the main actor "
            + "(`await MainActor.run { ... }` / `DispatchQueue.main.async`); only the runtime calls "
            + "(setBinding, trigger, setInputs, getBinding, currentState, ...) take any thread"
    }
}
