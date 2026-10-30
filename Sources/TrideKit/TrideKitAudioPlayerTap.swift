/// What the app's `AVPlayer` plays — a file or an endless stream such as internet radio — tapped for
/// a scene: made by `TrideKitScene.attachAudioPlayer(_:input:)`.
///
/// ## Public API
/// - `TrideKitAudioPlayerTap` — `input`, `player`, `isTapped` (the scene hears the current item);
///   `sendsSilence` (`@_spi(TrideTesting)`: the device gets zeros, the scene still hears the sound).
///
/// ## Non-obvious constraints
/// - An `MTAudioProcessingTap` on the current item's audio mix copies what the player processes — a
///   playback buffer before it is heard — into a ring (10 s, `AudioTapRing`) the scene's next step
///   hands to the runtime as a stream. The sound passes through untouched; nothing allocates there.
/// - The input's audio time is the item's: a sample is labelled with the item time it plays at, and
///   the scene hears the player's clock (`currentTime`, which AVPlayer keeps on what is heard, the
///   output latency included). Until the sound of that moment has arrived the scene hears silence.
/// - Labels count samples from the tap's stamps (`AudioTapTimeline`): a stamp is up to a hop (~12
///   ms; on macOS ~25 ms more) late, so it moves the count only back; a jump past 20 ms (a seek, a
///   loop, a resume) starts the stream there. Buffers covering no item time (the tap is called on
///   while the player is paused or seeking) are not the item's sound and are left out.
/// - Another item (another station) is tapped and starts the input afresh. The app keeps driving the
///   player — play, pause, seek, volume; the volume is the gain (0 while muted or paused).
/// - Protected items and HTTP Live Streams hand no audio to a tap: silence, and one diagnostic.
/// - A tap the app's own audio mix already carries on that track is replaced; its other settings
///   stay. Detached, or once the scene goes, the item gets its own audio mix back.

import AVFoundation
import Foundation
import MediaToolbox
import Observation
import os

#if canImport(TrideCore)
import TrideCore
#endif

/// The audio thread's side of one tapped item: the ring of what its tap copied, each buffer stamped
/// with the item time the tap gave it (`AudioTapRing`), made in the tap's `prepare` once the format
/// is known.
final class PlayerTapState: @unchecked Sendable {
    typealias Drained = AudioTapRing<Double>.Drained

    private struct Current {
        var ring: AudioTapRing<Double>?
        /// The tap met samples it cannot read (not 32-bit float).
        var unsupported = false
    }

    private let current = OSAllocatedUnfairLock(initialState: Current())
    private let silence: OSAllocatedUnfairLock<Bool>
    /// Seconds kept for a scene that does not step: the oldest go first. Longer than the engine's —
    /// a player processes its sound well before it is heard.
    private static let keepSeconds = 10.0

    init(sendsSilence: Bool) {
        silence = OSAllocatedUnfairLock(initialState: sendsSilence)
    }

    /// The device gets zeros instead of the item's sound; the scene still hears it.
    var sendsSilence: Bool {
        get { silence.withLock { $0 } }
        set { silence.withLock { $0 = newValue } }
    }

    /// The tap's processing format, from its `prepare` callback (not real time: the ring is made here).
    func prepare(_ asbd: AudioStreamBasicDescription) {
        let ring = AudioTapFormat(asbd).map { AudioTapRing<Double>(format: $0, keepSeconds: Self.keepSeconds) }
        let old = current.withLock { c -> AudioTapRing<Double>? in
            defer { c = Current(ring: ring, unsupported: ring == nil) }
            return c.ring
        }
        old?.close()
    }

    /// The audio thread: a buffer of `frames` frames the tap got for `range` of the item.
    func process(_ list: UnsafeMutablePointer<AudioBufferList>, frames: Int, range: CMTimeRange) {
        // A buffer that covers no item time is not the item's sound: the player is paused (the tap
        // keeps being called, its range frozen) or seeking.
        if range.start.isNumeric, range.duration.isNumeric, range.duration.value > 0 {
            let start = range.start.seconds
            // Unchecked: the system's buffer list is read only inside this callback, on this thread.
            current.withLockUnchecked { $0.ring?.capture(list, frames: frames, stamp: start, silent: false) }
        }
        if silence.withLock({ $0 }) {
            for buffer in UnsafeMutableAudioBufferListPointer(list) {
                if let data = buffer.mData { memset(data, 0, Int(buffer.mDataByteSize)) }
            }
        }
    }

    /// The step: the ring of the current format, and whether the tap met samples it cannot read.
    func take() -> (ring: AudioTapRing<Double>?, unsupported: Bool) {
        current.withLock { ($0.ring, $0.unsupported) }
    }

    /// Nothing is copied from now on (the item lost its tap).
    func close() {
        current.withLock { $0.ring }?.close()
    }
}

/// An `MTAudioProcessingTap` that copies what passes into `state` and leaves the sound as it is.
private func makeAudioTap(_ state: PlayerTapState) -> MTAudioProcessingTap? {
    // The tap owns one reference to the state; `finalize` gives it back.
    let info = UnsafeMutableRawPointer(Unmanaged.passRetained(state).toOpaque())
    var callbacks = MTAudioProcessingTapCallbacks(
        version: kMTAudioProcessingTapCallbacksVersion_0,
        clientInfo: info,
        init: { _, clientInfo, storage in
            storage.pointee = clientInfo
        },
        finalize: { tap in
            Unmanaged<PlayerTapState>.fromOpaque(MTAudioProcessingTapGetStorage(tap)).release()
        },
        prepare: { tap, _, format in
            Unmanaged<PlayerTapState>.fromOpaque(MTAudioProcessingTapGetStorage(tap))
                .takeUnretainedValue()
                .prepare(format.pointee)
        },
        unprepare: { _ in },
        process: { tap, frames, _, buffers, framesOut, flagsOut in
            var range = CMTimeRange()
            guard MTAudioProcessingTapGetSourceAudio(tap, frames, buffers, flagsOut, &range, framesOut) == noErr
            else { return }
            Unmanaged<PlayerTapState>.fromOpaque(MTAudioProcessingTapGetStorage(tap))
                .takeUnretainedValue()
                .process(buffers, frames: Int(framesOut.pointee), range: range)
        }
    )
    var tap: MTAudioProcessingTap?
    let status = MTAudioProcessingTapCreate(
        kCFAllocatorDefault, &callbacks, kMTAudioProcessingTapCreationFlag_PostEffects, &tap)
    guard status == noErr, let tap else {
        // No tap took the state: give back the reference kept for it.
        Unmanaged<PlayerTapState>.fromOpaque(info).release()
        return nil
    }
    return tap
}

/// The sound of an `AVPlayer` the app plays, heard by a scene.
@available(iOS 17.0, macCatalyst 17.0, *)
@MainActor
@Observable
public final class TrideKitAudioPlayerTap {
    /// The scene's input that hears the player (`audio.<input>.<probe>`).
    public let input: String
    /// The player — the app keeps driving it.
    public let player: AVPlayer
    /// `true` while the scene hears the player's current item; `false` without an item, while one
    /// loads, or for one whose sound cannot be tapped (a diagnostic says why).
    public private(set) var isTapped = false
    /// The device gets zeros instead of the player's sound while the scene still hears it: a test
    /// plays nothing aloud.
    @_spi(TrideTesting) @ObservationIgnored public var sendsSilence = false {
        didSet { state?.sendsSilence = sendsSilence }
    }

    /// The current item's tap.
    @ObservationIgnored private(set) var state: PlayerTapState?
    @ObservationIgnored private var ring: AudioTapRing<Double>?
    /// The buffers on the item's time. The tap stamps a buffer up to a hop of its processing
    /// (~12 ms) late, so a later stamp within 20 ms continues the count and an earlier one moves it
    /// back; a jump past 20 ms either way — a seek, a loop, a resume — starts the stream there.
    @ObservationIgnored private var timeline = AudioTapTimeline(slack: 0.02, jumpBack: 0.02)
    @ObservationIgnored private var sink: AudioSink?
    @ObservationIgnored private var observation: NSKeyValueObservation?
    @ObservationIgnored private var tappedItem: AVPlayerItem?
    @ObservationIgnored private var mixBefore: AVAudioMix?
    @ObservationIgnored private var tapping: Task<Void, Never>?
    /// The format the input was begun with — the tap's own once it reported one.
    @ObservationIgnored private var format = (sampleRate: 48_000.0, channels: 2)
    @ObservationIgnored private var saidUnsupported = false
    @ObservationIgnored private var released = false

    init(player: AVPlayer, input: String) {
        self.player = player
        self.input = input
    }

    private func itemChanged() {
        guard !released, player.currentItem !== tappedItem || !isTapped else { return }
        untap()
        // Another item — another station: what the input had is not its sound.
        sink?.reset()
        guard let item = player.currentItem else { return }
        tapping?.cancel()
        tapping = Task { [weak self] in await self?.tap(item) }
    }

    private func tap(_ item: AVPlayerItem) async {
        do {
            let asset = item.asset
            if try await asset.load(.hasProtectedContent) {
                sink?.problem("the player's item is protected — the scene cannot hear it and hears silence")
                return
            }
            let tracks = try await asset.loadTracks(withMediaType: .audio)
            guard !Task.isCancelled, !released, player.currentItem === item else { return }
            guard let track = tracks.first else {
                sink?.problem(
                    "the player's item hands no audio track to a tap (an HTTP Live Stream on this system) — the scene hears silence")
                return
            }
            let next = PlayerTapState(sendsSilence: sendsSilence)
            guard let processor = makeAudioTap(next) else {
                sink?.problem("the system refused an audio tap on the player — the scene hears silence")
                return
            }
            let own = item.audioMix?.inputParameters.first { $0.trackID == track.trackID }
            let parameters = (own?.mutableCopy() as? AVMutableAudioMixInputParameters)
                ?? AVMutableAudioMixInputParameters(track: track)
            parameters.audioTapProcessor = processor
            let mix = AVMutableAudioMix()
            mix.inputParameters = (item.audioMix?.inputParameters ?? []).filter { $0.trackID != track.trackID }
                + [parameters]
            mixBefore = item.audioMix
            item.audioMix = mix
            tappedItem = item
            state = next
            isTapped = true
        } catch {
            sink?.problem("the player's item could not be read: \(error.localizedDescription)")
        }
    }

    /// The item gets the audio mix it had before the tap; the input's count starts afresh.
    private func untap() {
        tapping?.cancel()
        tapping = nil
        if let item = tappedItem { item.audioMix = mixBefore }
        tappedItem = nil
        mixBefore = nil
        state?.close()
        state = nil
        ring = nil
        timeline.reset()
        isTapped = false
    }
}

@available(iOS 17.0, macCatalyst 17.0, *)
extension TrideKitAudioPlayerTap: SceneAudioSource {
    var kind: TrideAudioKind { .stream }
    var sampleRate: Double { format.sampleRate }
    var channels: Int { format.channels }

    func connect(_ sink: AudioSink) {
        self.sink = sink
        observation = player.observe(\.currentItem, options: [.initial, .new]) { [weak self] _, _ in
            Task { @MainActor in self?.itemChanged() }
        }
    }

    func pump() {
        guard let sink, let state else { return }
        let current = state.take()
        if current.unsupported, !saidUnsupported {
            saidUnsupported = true
            sink.problem("the player's audio is not 32-bit float samples — the scene hears silence")
        }
        guard let next = current.ring else { return }
        if next !== ring {
            // The tap (re)started in this format: a new one starts the input afresh.
            ring = next
            timeline.reset()
            if next.format.sampleRate != format.sampleRate || next.format.channels != format.channels {
                format = (next.format.sampleRate, next.format.channels)
                sink.begin(sampleRate: next.format.sampleRate, channels: next.format.channels, kind: .stream)
            }
        }
        let drained = next.drain()
        let heard = timeline.chunks(
            drained.pcm, channels: next.format.channels, rate: next.format.sampleRate,
            renders: drained.renders.map { ($0.offset, $0.stamp) })
        for chunk in heard.chunks { sink.append(startTime: chunk.start, pcm: chunk.pcm) }
    }

    func now() -> (time: Double, gain: Float) {
        // The player's clock runs at what is heard: AVPlayer keeps video in step with the sound
        // reaching the listener, output latency included. The tap gets a buffer well before (the
        // player's own buffering); until the sound of this moment has arrived — a moment after a
        // seek — the scene hears silence, never what is still to come.
        let time = player.currentTime()
        let seconds = time.isNumeric ? time.seconds : 0
        guard isTapped, player.timeControlStatus == .playing, let head = timeline.head, seconds < head else {
            return (seconds, 0)
        }
        return (seconds, player.isMuted ? 0 : player.volume)
    }

    func release() {
        released = true
        observation?.invalidate()
        observation = nil
        untap()
    }

    nonisolated func sceneGone() {
        // The scene went with the player still tapped: the item gets its own audio mix back, and
        // the next item is not tapped.
        Task { @MainActor in self.release() }
    }
}
