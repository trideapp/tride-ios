/// CADisplayLink-driven frame timer that clamps delta time before forwarding milliseconds into `TrideScene.tick`.
///
/// ## Public API
/// - `TickDriver` — start/stop display link and invoke a closure with stable `dtMs` each wake;
///   `preferredRate` — the host's render rate: the link wakes at most that often.
/// - `FramePacer` / `TickClock` — the wake rule and the step it hands the core, host-independent
///   (unit-tested on macOS, where the display link is a stub).
///
/// ## Non-obvious constraints
/// - Uses main runloop `.common` mode; caps dt at 33ms (two 60Hz frames) after measuring
///   elapsed time via consecutive `timestamp` values (not `targetTimestamp - timestamp`, which
///   is the nominal frame interval, not wall time between callbacks). With a render rate the cap is
///   two of its frames, so 30 wakes a second still hand the core every millisecond.
/// - A render rate never changes the scene's logic rate: the core steps a fixed 60 Hz over the time a
///   wake hands it.
///   Touches are queued in the core (`pointer`) and resolve on the next wake.
/// - `nil` = every display frame, exactly as before the render rate existed: the link's
///   `preferredFrameRateRange` is left untouched.

import Foundation
import QuartzCore
import os

private let logger = Logger(subsystem: "studio.tride.kit", category: "TickDriver")

/// The host's render rate as the views take it: `nil` keeps the default; a value of 0
/// or less is refused (the default stays, the view warns once). A rate above the screen's is the
/// screen's: the system caps the link and `FramePacer` never wakes twice in one display frame.
enum RenderRate {
    static func resolve(_ requested: Int?) -> (fps: Int?, refused: Bool) {
        guard let requested else { return (nil, false) }
        return requested > 0 ? (requested, false) : (nil, true)
    }

    static func refusalMessage(_ requested: Int) -> String {
        "preferredFPS \(requested) is not a frame rate (it must be above 0) — the view keeps its default rate"
    }
}

/// Which display callbacks wake the scene at a render rate: a wake is due once the time
/// since the last one reaches the rate's interval, less half a display frame — so 30 on a 60 Hz or
/// a 120 Hz screen wakes every second or fourth callback, steadily, through vsync jitter. The leftover
/// is kept (a rate that does not divide the screen's averages out) within half an interval: a stall
/// is not caught up with a burst of wakes.
struct FramePacer {
    /// Wakes a second; `nil` = every callback.
    private(set) var fps: Int?
    private var lastCallback: CFTimeInterval?
    private var budget: CFTimeInterval = 0

    init(fps: Int? = nil) {
        self.fps = fps
    }

    /// Seconds between wakes; `nil` = every callback.
    var interval: CFTimeInterval? { fps.map { 1.0 / CFTimeInterval($0) } }

    mutating func setRate(_ fps: Int?) {
        self.fps = fps
        budget = 0
    }

    mutating func reset() {
        lastCallback = nil
        budget = 0
    }

    /// `true` when the callback at `now` (seconds) wakes the scene.
    mutating func admit(_ now: CFTimeInterval) -> Bool {
        guard let interval else {
            lastCallback = now
            return true
        }
        guard let last = lastCallback else {
            lastCallback = now
            budget = 0
            return true
        }
        let frame = now - last
        guard frame > 0 else { return false }
        lastCallback = now
        budget += frame
        if budget < interval - frame / 2 { return false }
        budget = min(max(budget - interval, -interval / 2), interval / 2)
        return true
    }
}

/// The step a wake hands the core: wall time since the previous wake, capped at two frames of the
/// wake rate (two 60 Hz frames without one) so a stall is not one giant step. One tick per wake: the
/// core owns its 60 Hz steps.
struct TickClock {
    static let maxFrameMs: Float = 33.0

    var pacer = FramePacer()
    private var prevWake: CFTimeInterval?

    var maxStepMs: Float {
        guard let interval = pacer.interval else { return Self.maxFrameMs }
        return max(Self.maxFrameMs, Float(interval * 2000.0))
    }

    mutating func reset() {
        pacer.reset()
        prevWake = nil
    }

    /// The step in ms for a display callback at `now`, or `nil` when the callback does not wake the
    /// scene; `nominal` (the link's frame duration) stands in for the first wake's elapsed time.
    mutating func wake(at now: CFTimeInterval, nominal: CFTimeInterval) -> Float? {
        guard pacer.admit(now) else { return nil }
        let dt = prevWake.map { max(0, now - $0) } ?? max(0, nominal)
        prevWake = now
        return min(Float(dt * 1000.0), maxStepMs)
    }
}

#if os(iOS) || targetEnvironment(macCatalyst)

@available(iOS 17.0, macCatalyst 17.0, *)
@MainActor
final class TickDriver {

    /// Default per-frame ms cap: two frames at 60Hz (~33ms). Limits catch-up after stalls
    /// without collapsing long gaps into a single 16ms step (which used nominal frame interval).
    static let maxFrameMs: Float = TickClock.maxFrameMs

    private let onTick: (Float) -> Void
    /// `nonisolated(unsafe)` so `deinit` can invalidate it: a scene dropped while playing must not
    /// leave a display link firing for the rest of the process.
    nonisolated(unsafe) private var displayLink: CADisplayLink?
    /// Wake rule + wall-clock dt between wakes (`CADisplayLink.timestamp`, host time).
    private var clock = TickClock()

    /// The render rate the link wakes at; `nil` = every display frame.
    var preferredRate: Int? { clock.pacer.fps }

    init(onTick: @escaping (Float) -> Void) {
        self.onTick = onTick
    }

    deinit {
        // The link retains its proxy until invalidated; invalidate it on the run loop it was added to.
        guard let link = displayLink else { return }
        if Thread.isMainThread {
            link.invalidate()
        } else {
            nonisolated(unsafe) let pending = link
            DispatchQueue.main.async { pending.invalidate() }
        }
    }

    func start() {
        guard displayLink == nil else { return }
        let proxy = DisplayLinkProxy { [weak self] link in
            self?.handle(link)
        }
        let link = CADisplayLink(
            target: proxy,
            selector: #selector(DisplayLinkProxy.tick(_:))
        )
        if let fps = clock.pacer.fps { link.preferredFrameRateRange = Self.range(fps) }
        link.add(to: .main, forMode: .common)
        displayLink = link
        logger.debug("CADisplayLink started")
    }

    func stop() {
        displayLink?.invalidate()
        displayLink = nil
        clock.reset()
        logger.debug("CADisplayLink stopped")
    }

    /// Wake at most `fps` times a second from the next display frame (`nil`: every frame). The step
    /// stays measured from the last wake, so a change mid-run loses no time.
    func setPreferredRate(_ fps: Int?) {
        guard fps != clock.pacer.fps else { return }
        let wasPaced = clock.pacer.fps != nil
        clock.pacer.setRate(fps)
        guard let link = displayLink else { return }
        if let fps {
            link.preferredFrameRateRange = Self.range(fps)
        } else if wasPaced {
            link.preferredFrameRateRange = .default
        }
    }

    private static func range(_ fps: Int) -> CAFrameRateRange {
        let rate = Float(fps)
        return CAFrameRateRange(minimum: rate, maximum: rate, preferred: rate)
    }

    private func handle(_ link: CADisplayLink) {
        // The first wake has no previous one: the link's frame duration stands in.
        guard let dtMs = clock.wake(at: link.timestamp, nominal: link.targetTimestamp - link.timestamp) else { return }
        onTick(dtMs)
    }

    /// `CADisplayLink` accepts only Objective-C selectors; this proxy
    /// adapts the Swift closure into one. Living separately from
    /// `TickDriver` keeps the driver safely deallocate-able.
    private final class DisplayLinkProxy: NSObject {
        let handler: (CADisplayLink) -> Void
        init(handler: @escaping (CADisplayLink) -> Void) { self.handler = handler }
        @objc func tick(_ link: CADisplayLink) { handler(link) }
    }
}

#else

@MainActor
final class TickDriver {
    static let maxFrameMs: Float = TickClock.maxFrameMs

    private let onTick: (Float) -> Void
    private var pacer = FramePacer()

    var preferredRate: Int? { pacer.fps }

    init(onTick: @escaping (Float) -> Void) {
        self.onTick = onTick
    }

    func setPreferredRate(_ fps: Int?) {
        pacer.setRate(fps)
    }

    func start() {
        logger.debug("TickDriver start ignored on macOS host tests")
    }

    func stop() {
        logger.debug("TickDriver stop ignored on macOS host tests")
    }
}

#endif
