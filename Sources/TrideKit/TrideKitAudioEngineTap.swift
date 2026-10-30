/// What the app's own `AVAudioEngine` plays, heard by a scene: made by
/// `TrideKitScene.attachAudioEngine(_:node:input:)`.
///
/// ## Public API
/// - `TrideKitAudioEngineTap` — `input`, `engine`, `node`, `isTapped` (the scene hears the node).
///
/// ## Non-obvious constraints
/// - A render notification on the node's own audio unit copies every render of its output (one IO
///   cycle, 5–20 ms) with the cycle's host time into a ring (2 s, `AudioTapRing` — the player tap's
///   too) the scene's next step hands the runtime as a stream. `installTap` hands 100 ms blocks ~80 ms after their first sample
///   (measured): later than the listener hears them. The notification takes no tap slot, so the
///   app's own `installTap` on the same node still works.
/// - The input's audio time is the host clock (seconds of `mach_absolute_time`) at which a sample is
///   heard: a render's host time (when it leaves for the device, already one IO buffer ahead of
///   the render) + the output latency. The step hears the host clock read before its drain, so it
///   has every sample heard by then, however long the main thread is held. A render off the
///   counted samples by more than 1 ms moves the count (`AudioTapTimeline`): forward leaves a gap,
///   backward drops the overlap (the runtime's stream never steps back).
/// - Silence: the engine not running, no render for 0.1 s (a stopped node, an interruption), a
///   detach. What the node outputs is the sound — the app's volume and effects before it included;
///   the system volume and the silent switch are not.
/// - Never the microphone: an input node is refused, and `engine.inputNode` is never touched.
/// - An engine configuration change, or a new format on the node, re-installs the notification and
///   starts the input afresh. The scene's `deinit` removes it on any thread; a render already
///   running may still call it, so its state lives one more second.

import AudioToolbox
import AVFoundation
import Foundation
import Observation
import os

#if canImport(TrideCore)
import TrideCore
#endif

/// The audio thread's side of an engine tap: the ring of what the node rendered, stamped with each
/// render's host time (`AudioTapRing`), and the render notification that feeds it.
final class EngineTapState: @unchecked Sendable {
    typealias Format = AudioTapFormat
    typealias Drained = AudioTapRing<UInt64>.Drained

    /// What `uninstall` needs: the unit (kept alive by its node) and the reference it was given.
    private struct Installed: @unchecked Sendable {
        let node: AVAudioNode
        let unit: AudioUnit
        let refCon: UnsafeMutableRawPointer
    }

    let format: Format
    private let ring: AudioTapRing<UInt64>
    private let installed = OSAllocatedUnfairLock<Installed?>(initialState: nil)
    /// Seconds kept for a scene that does not step (paused, in the background): the oldest go first.
    private static let keepSeconds = 2.0

    init(format: Format) {
        self.format = format
        ring = AudioTapRing(format: format, keepSeconds: Self.keepSeconds)
    }

    /// `true` while the render notification sits on the node's unit.
    var isInstalled: Bool { installed.withLock { $0 != nil } }

    /// Put the notification on `node`'s audio unit: `noErr`, or why not (no unit, a refusal).
    func install(on node: AVAudioNode) -> OSStatus {
        guard let unit = renderUnit(of: node) else { return kAudioUnitErr_InvalidElement }
        // The notification owns one reference to the state; `uninstall` gives it back.
        let refCon = Unmanaged.passRetained(self).toOpaque()
        let status = AudioUnitAddRenderNotify(unit, engineRenderNotify, refCon)
        guard status == noErr else {
            Unmanaged<EngineTapState>.fromOpaque(refCon).release()
            return status
        }
        let next = Installed(node: node, unit: unit, refCon: refCon)
        installed.withLock { $0 = next }
        return noErr
    }

    /// Take the notification off (any thread, once); nothing is copied from now on.
    func uninstall() {
        ring.close()
        guard let gone = installed.withLock({ state -> Installed? in
            defer { state = nil }
            return state
        }) else { return }
        AudioUnitRemoveRenderNotify(gone.unit, engineRenderNotify, gone.refCon)
        // A render running now may still call the notification: its reference goes a second later.
        let late = LateRelease(refCon: gone.refCon)
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 1) { late.release() }
    }

    /// The audio thread: a render of `frames` frames left the node with `time`.
    func capture(_ list: UnsafeMutablePointer<AudioBufferList>, frames: Int, time: AudioTimeStamp, silent: Bool) {
        let host = time.mFlags.contains(.hostTimeValid) && time.mHostTime != 0 ? time.mHostTime : nil
        ring.capture(list, frames: frames, stamp: host, silent: silent)
    }

    /// The step: everything rendered since the last drain that is still kept.
    func drain() -> Drained {
        ring.drain()
    }
}

/// A reference handed to a render notification, given back later on another thread.
private struct LateRelease: @unchecked Sendable {
    let refCon: UnsafeMutableRawPointer
    func release() { Unmanaged<EngineTapState>.fromOpaque(refCon).release() }
}

/// The audio unit that renders `node`'s output: an effect / instrument's own, the I/O unit, or the
/// unit behind a mixer or a player node.
private func renderUnit(of node: AVAudioNode) -> AudioUnit? {
    if let unit = node as? AVAudioUnit { return unit.audioUnit }
    if let io = node as? AVAudioIONode { return io.audioUnit }
    return (node.auAudioUnit as? AUAudioUnitV2Bridge)?.audioUnit
}

/// After each render of output bus 0: copy what it rendered (a failed render is skipped).
private let engineRenderNotify: AURenderCallback = { refCon, flags, time, bus, frames, data in
    let action = flags.pointee
    guard action.contains(.unitRenderAction_PostRender), !action.contains(.unitRenderAction_PostRenderError),
          bus == 0, let data
    else { return noErr }
    Unmanaged<EngineTapState>.fromOpaque(refCon).takeUnretainedValue().capture(
        data, frames: Int(frames), time: time.pointee, silent: action.contains(.unitRenderAction_OutputIsSilence))
    return noErr
}

/// The sound of an `AVAudioEngine` node the app plays, heard by a scene.
@available(iOS 17.0, macCatalyst 17.0, *)
@MainActor
@Observable
public final class TrideKitAudioEngineTap {
    /// The scene's input that hears the node (`audio.<input>.<probe>`).
    public let input: String
    /// The engine — the app keeps driving it.
    public let engine: AVAudioEngine
    /// The node whose output the scene hears.
    public let node: AVAudioNode
    /// `true` while the scene hears the node; `false` after a detach, or when its audio unit
    /// refused the tap (a diagnostic says why).
    public private(set) var isTapped = false

    /// `nonisolated(unsafe)` so `deinit` and `sceneGone` can take the notification off: every
    /// other access is on the main actor.
    @ObservationIgnored nonisolated(unsafe) private(set) var state: EngineTapState?
    /// The format the input was begun with: the node's at the last install.
    @ObservationIgnored private(set) var sampleRate: Double
    @ObservationIgnored private(set) var channels: Int
    @ObservationIgnored private var sink: AudioSink?
    @ObservationIgnored nonisolated(unsafe) private var observer: NSObjectProtocol?
    @ObservationIgnored private var released = false
    /// The renders on the host clock: a render off the count by more than 1 ms moves it.
    @ObservationIgnored private var timeline = AudioTapTimeline()
    @ObservationIgnored private var saidNoHostTime = false
    /// The host clock as the step's `pump` began: the moment the step hears.
    @ObservationIgnored private var pumpedAt: Double?

    /// No render for this long is silence (the engine or the node stopped).
    private static let stale = 0.1

    private static func hostNow() -> Double { AVAudioTime.seconds(forHostTime: mach_absolute_time()) }

    init(engine: AVAudioEngine, node: AVAudioNode, input: String) throws {
        if node is AVAudioInputNode {
            throw TrideError.InvalidAudio(message: "audio input \(input): the scene never hears the microphone")
        }
        guard node.engine === engine else {
            throw TrideError.InvalidAudio(message: "audio input \(input): the node is not attached to this engine")
        }
        guard let format = EngineTapState.Format(node.outputFormat(forBus: 0)) else {
            throw TrideError.InvalidAudio(
                message: "audio input \(input): the node's output is not 32-bit float samples")
        }
        self.engine = engine
        self.node = node
        self.input = input
        sampleRate = format.sampleRate
        channels = format.channels
    }

    deinit {
        state?.uninstall()
        if let observer { NotificationCenter.default.removeObserver(observer) }
    }

    /// The notification on the node, with the format it renders now; the input starts afresh.
    private func install() {
        state?.uninstall()
        state = nil
        timeline.reset()
        isTapped = false
        guard !released, let sink else { return }
        guard let format = EngineTapState.Format(node.outputFormat(forBus: 0)) else {
            sink.problem("the node's output is not 32-bit float samples — the scene hears silence")
            return
        }
        if format.sampleRate == sampleRate, format.channels == channels {
            sink.reset()
        } else {
            sampleRate = format.sampleRate
            channels = format.channels
            sink.begin(sampleRate: format.sampleRate, channels: format.channels, kind: .stream)
        }
        let next = EngineTapState(format: format)
        let status = next.install(on: node)
        guard status == noErr else {
            sink.problem("the node's audio unit refused a tap (\(status)) — the scene hears silence")
            return
        }
        state = next
        isTapped = true
    }

}

@available(iOS 17.0, macCatalyst 17.0, *)
extension TrideKitAudioEngineTap: SceneAudioSource {
    var kind: TrideAudioKind { .stream }

    func connect(_ sink: AudioSink) {
        self.sink = sink
        // The system stops the engine and may rebuild its graph (a new route, a new rate).
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.install() }
        }
        install()
    }

    func pump() {
        // The clock before the drain: everything heard by then has been rendered, so the drain has it.
        pumpedAt = Self.hostNow()
        guard let sink, !released else { return }
        // The app connected the node anew in another format: start again in it.
        if let state, EngineTapState.Format(node.outputFormat(forBus: 0)) != state.format { install() }
        guard let state else { return }
        let drained = state.drain()
        // A render leaves for the device at its host time; the listener hears it the output's
        // latency later (the speaker, a Bluetooth link). No host time: it follows the one before.
        let latency = audioOutputLatency(of: engine)
        let renders = drained.renders.map { render in
            (render.offset, render.stamp.map { AVAudioTime.seconds(forHostTime: $0) + latency })
        }
        let heard = timeline.chunks(
            drained.pcm, channels: state.format.channels, rate: state.format.sampleRate, renders: renders)
        if heard.untimed, !saidNoHostTime {
            saidNoHostTime = true
            sink.problem("the engine renders without a host time (manual rendering?) — the scene hears silence")
        }
        for chunk in heard.chunks { sink.append(startTime: chunk.start, pcm: chunk.pcm) }
    }

    func now() -> (time: Double, gain: Float) {
        let time = pumpedAt ?? Self.hostNow()
        pumpedAt = nil
        guard isTapped, engine.isRunning, let head = timeline.head, time - head < Self.stale else { return (time, 0) }
        return (time, 1)
    }

    func release() {
        released = true
        if let observer { NotificationCenter.default.removeObserver(observer) }
        observer = nil
        state?.uninstall()
        state = nil
        timeline.reset()
        isTapped = false
    }

    nonisolated func sceneGone() {
        state?.uninstall()
    }
}
