/// A music track the app plays through TrideKit and a scene hears whole — made by
/// `TrideKitScene.attachAudioTrack(url:input:loop:)`.
///
/// ## Public API
/// - `TrideKitAudioTrack` — `play()` / `pause()` / `seek(to:)`, `currentTime`, `duration`, `isPlaying`,
///   `loop`, `volume`, `isMuted`; `input`, `url`.
///
/// ## Non-obvious constraints
/// - Decoded by the platform (`AVAudioFile`: whatever it reads — mp3, AAC, ALAC, WAV, …). The scene
///   hears the first two channels at the file's own rate, fed to the runtime on a background
///   queue in one-second slices: the runtime analyses the whole track far faster than it plays.
/// - Played by its own `AVAudioEngine` → `AVAudioPlayerNode`, straight from the file (nothing held in
///   memory). Nothing records: the engine's input is never touched, and the app's audio session
///   (category, activation) stays the app's.
/// - What the scene hears every step: the player's position − the output latency, at the track's
///   volume — 0 while muted, paused, ended, or while the system holds the engine (an interruption).
///   The position is the sample leaving the engine now: the node's last render says which sample
///   leaves at which host time (an IO buffer or two ahead), so it moves smoothly between renders;
///   the session's `outputLatency` covers the rest (the speaker, a Bluetooth link). Without a host
///   time the render counts one IO buffer ahead.
/// - A configuration change (a new output route) restarts a playing track where it was; so does the
///   end of an interruption the system says may resume.
/// - `fed()` (`@_spi(TrideTesting)`) waits for the whole track to reach the runtime.
/// - Main actor only, like the scene.

import AVFoundation
import Foundation
import Observation
import os

#if canImport(TrideCore)
import TrideCore
#endif

private let logger = Logger(subsystem: "studio.tride.kit", category: "Audio")

/// A track the app controls and the scene hears.
@available(iOS 17.0, macCatalyst 17.0, *)
@MainActor
@Observable
public final class TrideKitAudioTrack {
    /// The scene's input that hears the track (`audio.<input>.<probe>`).
    public let input: String
    /// The file.
    public let url: URL
    /// Length in seconds.
    public let duration: TimeInterval
    /// `true` from `play()` until `pause()`, the end of a track that does not loop, or a detach.
    public private(set) var isPlaying = false

    /// Start again at the end.
    public var loop: Bool {
        get { looping }
        set {
            guard newValue != looping else { return }
            looping = newValue
            // The queued tail follows the new rule.
            if isPlaying { restart(at: currentTime) }
        }
    }

    /// Playback volume, 0 and up (1 = as decoded). The scene hears it too.
    public var volume: Float {
        get { level }
        set {
            level = newValue.isFinite ? max(0, newValue) : level
            applyVolume()
        }
    }

    /// Silent for the listener, and for the scene.
    public var isMuted: Bool {
        get { muted }
        set {
            muted = newValue
            applyVolume()
        }
    }

    /// Playback position in seconds.
    public var currentTime: TimeInterval { position() }

    private var looping: Bool
    private var level: Float = 1
    private var muted = false

    @ObservationIgnored let sampleRate: Double
    @ObservationIgnored let channels: Int
    @ObservationIgnored private let file: AVAudioFile
    @ObservationIgnored private let length: AVAudioFramePosition
    /// `nonisolated(unsafe)` (with `observers`) so `deinit` can stop it: every other access is on the
    /// main actor.
    @ObservationIgnored nonisolated(unsafe) private let engine = AVAudioEngine()
    @ObservationIgnored private let node = AVAudioPlayerNode()
    /// The file frame the scheduled run starts at.
    @ObservationIgnored private var runStart: AVAudioFramePosition = 0
    /// Position while not playing.
    @ObservationIgnored private var offset: TimeInterval = 0
    /// Bumped by every (re)schedule: a completion of an older one is ignored.
    @ObservationIgnored private var generation = 0
    /// The position the node last reported — where a run the system stopped resumes.
    @ObservationIgnored private var lastPosition: TimeInterval = 0
    @ObservationIgnored private var feed: Task<Void, Never>?
    @ObservationIgnored nonisolated(unsafe) private var observers: [NSObjectProtocol] = []
    @ObservationIgnored private var released = false

    init(url: URL, input: String, loop: Bool) throws {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        self.input = input
        self.url = url
        self.file = file
        self.length = max(0, file.length)
        self.sampleRate = format.sampleRate
        self.channels = min(2, max(1, Int(format.channelCount)))
        self.duration = format.sampleRate > 0 ? Double(length) / format.sampleRate : 0
        self.looping = loop
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        applyVolume()
        observe()
    }

    deinit {
        feed?.cancel()
        engine.stop()
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
    }

    /// Start or resume playback (an ended track starts again). Throws when the system refuses to start
    /// the audio engine.
    public func play() throws {
        guard !isPlaying, !released else { return }
        if !looping && offset >= duration { offset = 0 }
        schedule(from: offset)
        if !engine.isRunning { try engine.start() }
        node.play()
        isPlaying = true
    }

    /// Hold the position; `play()` resumes there.
    public func pause() {
        guard isPlaying else { return }
        offset = position()
        isPlaying = false
        generation += 1
        node.stop()
        engine.pause()
    }

    /// Jump to `time` seconds, playing or not.
    public func seek(to time: TimeInterval) {
        let at = time.isFinite ? min(max(0, time), duration) : 0
        if isPlaying { restart(at: at) } else { offset = at }
    }

    /// Feed the whole track to `sink`, a slice at a time, off the main thread.
    func feed(into sink: AudioSink) {
        feed?.cancel()
        let url = url
        let channels = channels
        feed = Task.detached(priority: .utility) {
            do {
                // Its own reader: the player node reads the other one.
                let reader = try AVAudioFile(forReading: url)
                let format = reader.processingFormat
                let slice = AVAudioFrameCount(max(1, format.sampleRate.rounded()))
                guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: slice) else {
                    sink.problem("the track's format cannot be read into samples")
                    return
                }
                // Reading past the end is an error, not an empty buffer: read what is left.
                while !Task.isCancelled, reader.framePosition < reader.length {
                    let start = reader.framePosition
                    let rest = AVAudioFrameCount(clamping: reader.length - start)
                    try reader.read(into: buffer, frameCount: min(slice, rest))
                    if buffer.frameLength == 0 { break }
                    sink.append(
                        startTime: Double(start) / format.sampleRate,
                        pcm: interleavedPCM(buffer, channels: channels))
                }
                if !Task.isCancelled { sink.end() }
            } catch {
                sink.problem("the track could not be decoded: \(error.localizedDescription)")
            }
        }
    }

    /// The whole track reached the runtime (or its feed stopped) — a test plays it only then, so
    /// every step reads analysed frames.
    @_spi(TrideTesting) public func fed() async {
        await feed?.value
    }

    /// What the scene hears now: the audible time and the gain. Right after `play()` the time is
    /// below the run's start for one output latency — nothing of the run is heard yet.
    func now() -> (time: Double, gain: Float) {
        guard isPlaying, engine.isRunning, duration > 0 else { return (position(), 0) }
        let played = Double(runStart) / sampleRate + (leftSeconds() ?? 0)
        var audible = played - audioOutputLatency(of: engine)
        if looping, audible >= 0 {
            audible = audible.truncatingRemainder(dividingBy: duration)
        } else if !looping {
            audible = min(audible, duration)
        }
        return (audible, muted ? 0 : level)
    }

    /// The scene no longer hears it: playback stops for good.
    func release() {
        released = true
        feed?.cancel()
        if isPlaying { pause() }
        engine.stop()
    }

    // MARK: - Playback

    /// Seconds of the current run that have left the engine by now (negative before its first
    /// sample does). The last render's sample time leaves at its host time — just ahead of now.
    private func leftSeconds() -> TimeInterval? {
        guard let renderTime = node.lastRenderTime, renderTime.isSampleTimeValid,
              let playerTime = node.playerTime(forNodeTime: renderTime), playerTime.sampleRate > 0
        else { return nil }
        let rendered = Double(playerTime.sampleTime) / playerTime.sampleRate
        guard renderTime.isHostTimeValid else { return rendered - audioIOBuffer() }
        let ahead = AVAudioTime.seconds(forHostTime: renderTime.hostTime)
            - AVAudioTime.seconds(forHostTime: mach_absolute_time())
        // A stale or odd stamp moves the time by a few renders at most.
        return rendered - min(max(ahead, -0.1), 0.5)
    }

    private func position() -> TimeInterval {
        guard isPlaying else { return offset }
        guard let left = leftSeconds() else { return lastPosition }
        let played = max(0, left)
        let t = Double(runStart) / sampleRate + played
        guard duration > 0 else { return 0 }
        lastPosition = looping ? t.truncatingRemainder(dividingBy: duration) : min(t, duration)
        return lastPosition
    }

    /// Play on from `time`. A node may only play on a running engine: one the system stopped (an
    /// interruption) is started again, or the track waits for `resume()`.
    private func restart(at time: TimeInterval) {
        schedule(from: time)
        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                return
            }
        }
        node.play()
    }

    /// Schedule from `time`: the rest of the file, and — looping — one whole run queued behind it,
    /// renewed as each is taken, so the loop has no gap.
    private func schedule(from time: TimeInterval) {
        generation += 1
        let run = generation
        node.stop()
        runStart = min(max(0, AVAudioFramePosition((time * sampleRate).rounded())), length)
        lastPosition = Double(runStart) / sampleRate
        let rest = AVAudioFrameCount(clamping: length - runStart)
        if looping {
            if rest > 0 {
                node.scheduleSegment(file, startingFrame: runStart, frameCount: rest, at: nil,
                                     completionCallbackType: .dataConsumed) { [weak self] _ in
                    Task { @MainActor in self?.queueLoop(run) }
                }
            } else {
                queueLoop(run)
            }
            queueLoop(run)
        } else if rest > 0 {
            node.scheduleSegment(file, startingFrame: runStart, frameCount: rest, at: nil,
                                 completionCallbackType: .dataPlayedBack) { [weak self] _ in
                Task { @MainActor in self?.ended(run) }
            }
        } else {
            Task { @MainActor in self.ended(run) }
        }
    }

    /// One more whole run behind the queued one.
    private func queueLoop(_ run: Int) {
        guard run == generation, looping, length > 0 else { return }
        node.scheduleSegment(file, startingFrame: 0, frameCount: AVAudioFrameCount(clamping: length), at: nil,
                             completionCallbackType: .dataConsumed) { [weak self] _ in
            Task { @MainActor in self?.queueLoop(run) }
        }
    }

    /// The run of a track that does not loop played to its end.
    private func ended(_ run: Int) {
        guard run == generation, isPlaying, !looping else { return }
        offset = duration
        isPlaying = false
        generation += 1
        node.stop()
        engine.pause()
    }

    private func applyVolume() {
        node.volume = muted ? 0 : level
    }

    /// The engine stops itself on a configuration change (a new output route) and, on iOS, on an
    /// interruption: a playing track starts again where it was once the system lets it.
    private func observe() {
        let center = NotificationCenter.default
        observers.append(center.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.resume() }
        })
        #if os(iOS)
        observers.append(center.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: .main
        ) { [weak self] note in
            // Resume only when the system says the interruption allows it (a call ended, not
            // another app taking the sound).
            let info = note.userInfo
            let type = (info?[AVAudioSessionInterruptionTypeKey] as? UInt).flatMap(AVAudioSession.InterruptionType.init)
            let options = AVAudioSession.InterruptionOptions(
                rawValue: info?[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0)
            guard type == .ended, options.contains(.shouldResume) else { return }
            MainActor.assumeIsolated { self?.resume() }
        })
        #endif
    }

    private func resume() {
        guard isPlaying, !released, !engine.isRunning else { return }
        do {
            schedule(from: lastPosition)
            try engine.start()
            node.play()
        } catch {
            logger.error("audio \(self.input, privacy: .public): the engine did not start again: \(error.localizedDescription, privacy: .public)")
        }
    }
}

@available(iOS 17.0, macCatalyst 17.0, *)
extension TrideKitAudioTrack: SceneAudioSource {
    var kind: TrideAudioKind { .track }

    func connect(_ sink: AudioSink) {
        feed(into: sink)
    }

    func pump() {}
}
