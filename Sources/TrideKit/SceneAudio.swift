/// The sound a scene hears: the app's audio sources by input name, fed to the runtime through its
/// audio contract and timed before every step. The analysis is the runtime's — nothing here reads
/// the samples beyond copying them.
///
/// ## Internal API
/// - `SceneAudioSource` — what the app plays: kind and format, the PCM it feeds a sink, what sounds
///   now. `TrideKitAudioTrack`, `TrideKitAudioPlayerTap` and `TrideKitAudioEngineTap`.
/// - `AudioSink` — where a source's PCM goes: one input of one runtime; any thread, closed on detach.
/// - `SceneAudioInputs` — the inputs of one scene: `attach` / `detach` / `hear` (before each step).
/// - `interleavedPCM(_:channels:)` — a decoded buffer → the interleaved `Float` PCM the runtime takes.
/// - `audioOutputLatency(of:)` / `audioIOBuffer()` — how long a sample leaving the engine takes to
///   reach the listener; how far a render runs ahead of the output.
///
/// ## Non-obvious constraints
/// - The app owns time: `hear` writes each input's audible time and gain before the step reads
///   them; the runtime reads no clock.
/// - A closed sink drops what still reaches it: `close` waits for an append in flight, so the
///   `resetAudio` of a detach always comes after the last append of the feed it ends.
/// - The runtime hears at most two channels (left, right); a mono source is both.
/// - A failing runtime call becomes a diagnostic (`audio.<input>`), never a throw into the step.

import AVFoundation
import Foundation

#if canImport(TrideCore)
import TrideCore
#endif

/// A sound the app plays and a scene hears.
@MainActor
protocol SceneAudioSource: AnyObject {
    var kind: TrideAudioKind { get }
    /// The PCM's own rate (Hz).
    var sampleRate: Double { get }
    /// Channels fed to the runtime: 1 or 2.
    var channels: Int { get }
    /// Start feeding `sink`.
    func connect(_ sink: AudioSink)
    /// Before each step: hand over what arrived since the last one (a stream's chunks).
    func pump()
    /// What sounds now: the audible time on the source's audio time, and the gain (0 = silent).
    func now() -> (time: Double, gain: Float)
    /// The scene no longer hears it: a track stops, a tap leaves the player.
    func release()
    /// The scene went (its `deinit`, any thread): stop what keeps running for it on the audio side.
    nonisolated func sceneGone()
}

extension SceneAudioSource {
    /// Most sources stop with their last reference.
    nonisolated func sceneGone() {}
}

/// One input of one runtime, for a source's feed on any thread.
final class AudioSink: @unchecked Sendable {
    let input: String
    private let lock = NSLock()
    /// `nil` once closed.
    private var scene: TrideScene?
    private let report: @Sendable (TrideKitDiagnostic) -> Void

    init(input: String, scene: TrideScene, report: @escaping @Sendable (TrideKitDiagnostic) -> Void) {
        self.input = input
        self.scene = scene
        self.report = report
    }

    /// Interleaved PCM whose first sample sounds at `startTime` of the input's audio time.
    func append(startTime: Double, pcm: [Float]) {
        guard !pcm.isEmpty else { return }
        call("append") { try $0.appendAudio(input: input, startTimeS: startTime, pcm: pcm) }
    }

    /// A track is complete: silence follows its tail.
    func end() {
        call("end") { try $0.endAudio(input: input) }
    }

    /// Start again in another format (a stream whose real format became known); drops what it had.
    func begin(sampleRate: Double, channels: Int, kind: TrideAudioKind) {
        call("begin") {
            try $0.beginAudio(
                input: input, sampleRate: UInt32(sampleRate.rounded()), channels: UInt16(channels), kind: kind)
        }
    }

    /// Drop what the input was given (another station, a jump in a stream's time).
    func reset() {
        call("reset") { try $0.resetAudio(input: input) }
    }

    /// Say something about this input's sound once (the scene dedupes).
    func problem(_ message: String) {
        report(TrideKitDiagnostic(kind: .load, subject: "audio.\(input)", message: message))
    }

    /// Nothing reaches the runtime from now on.
    func close() {
        lock.withLock { scene = nil }
    }

    private func call(_ what: String, _ body: (TrideScene) throws -> Void) {
        lock.withLock {
            guard let scene else { return }
            do {
                try body(scene)
            } catch {
                report(TrideKitDiagnostic(
                    kind: .runtime, subject: "audio.\(input)", message: "\(what) failed: \(error.localizedDescription)"))
            }
        }
    }
}

/// The inputs one scene hears.
@MainActor
final class SceneAudioInputs {
    private struct Entry {
        let input: String
        let source: SceneAudioSource
        let sink: AudioSink
    }

    /// `nonisolated(unsafe)` so `deinit` can close the sinks: every other access is on the main actor.
    nonisolated(unsafe) private var entries: [Entry] = []
    /// What each input sounded like at the last step: the audible time and gain written.
    private(set) var heard: [String: (time: Double, gain: Float)] = [:]

    deinit {
        // The scene went: a feed still running reaches nothing (the sinks are thread-safe), and a
        // tap on the app's audio comes off even while the app holds it.
        for entry in entries {
            entry.sink.close()
            entry.source.sceneGone()
        }
    }

    /// The inputs heard now, in the order they were attached.
    var names: [String] { entries.map(\.input) }

    /// The scene hears `source` as `input` — one already there is detached first. Throws when the
    /// runtime refuses the input (a name that is no slug); nothing is attached then.
    func attach(
        _ input: String,
        _ source: SceneAudioSource,
        to scene: TrideScene,
        report: @escaping @Sendable (TrideKitDiagnostic) -> Void
    ) throws {
        detach(input, from: scene, report: report)
        try scene.beginAudio(
            input: input,
            sampleRate: UInt32(source.sampleRate.rounded()),
            channels: UInt16(source.channels),
            kind: source.kind
        )
        let sink = AudioSink(input: input, scene: scene, report: report)
        entries.append(Entry(input: input, source: source, sink: sink))
        source.connect(sink)
    }

    /// The scene stops hearing `input`; its source is released. `false`: no such input.
    @discardableResult
    func detach(_ input: String, from scene: TrideScene, report: (TrideKitDiagnostic) -> Void) -> Bool {
        guard let index = entries.firstIndex(where: { $0.input == input }) else { return false }
        let entry = entries.remove(at: index)
        entry.sink.close()
        entry.source.release()
        heard[input] = nil
        do {
            try scene.resetAudio(input: input)
        } catch {
            report(TrideKitDiagnostic(
                kind: .runtime, subject: "audio.\(input)", message: "reset failed: \(error.localizedDescription)"))
        }
        return true
    }

    /// Before each step: every input's new PCM, then what it sounds like now.
    func hear(_ scene: TrideScene, report: (TrideKitDiagnostic) -> Void) {
        for entry in entries {
            entry.source.pump()
            let now = entry.source.now()
            let time = now.time.isFinite ? now.time : 0
            let gain = now.gain.isFinite ? max(0, now.gain) : 0
            do {
                try scene.setAudioTime(input: entry.input, audibleTimeS: time, gain: gain)
                heard[entry.input] = (time, gain)
            } catch {
                report(TrideKitDiagnostic(
                    kind: .runtime, subject: "audio.\(entry.input)", message: "time failed: \(error.localizedDescription)"))
            }
        }
    }
}

/// The first `channels` channels of `buffer` (Float32, interleaved or not), interleaved — what the
/// runtime's `appendAudio` takes. A mono buffer read as stereo repeats its one channel.
func interleavedPCM(_ buffer: AVAudioPCMBuffer, channels: Int) -> [Float] {
    let frames = Int(buffer.frameLength)
    let available = Int(buffer.format.channelCount)
    guard frames > 0, channels > 0, available > 0, let data = buffer.floatChannelData else { return [] }
    var out = [Float](repeating: 0, count: frames * channels)
    if buffer.format.isInterleaved {
        // One buffer, `available` samples a frame.
        let src = data[0]
        for frame in 0..<frames {
            for c in 0..<channels {
                out[frame * channels + c] = src[frame * available + min(c, available - 1)]
            }
        }
    } else {
        for c in 0..<channels {
            let src = data[min(c, available - 1)]
            for frame in 0..<frames {
                out[frame * channels + c] = src[frame]
            }
        }
    }
    return out
}

/// How long a sample leaving the engine takes to reach the listener: the output's own latency
/// (the speaker, a Bluetooth link).
@MainActor
func audioOutputLatency(of engine: AVAudioEngine) -> TimeInterval {
    #if os(iOS)
    return AVAudioSession.sharedInstance().outputLatency
    #else
    return engine.outputNode.presentationLatency
    #endif
}

/// One IO buffer: how far a render runs ahead of the output when its host time is unknown.
@MainActor
func audioIOBuffer() -> TimeInterval {
    #if os(iOS)
    return AVAudioSession.sharedInstance().ioBufferDuration
    #else
    return 0
    #endif
}
