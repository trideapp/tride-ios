/// Sound the scene hears: the app plays music and the scene moves to it — its Sound effectors and
/// Sound fields, and any property bound to `audio.<input>.<probe>` (bloom strength, a scale, a colour).
///
/// ## Public API
/// - `attachAudioTrack(url:input:loop:)` → `TrideKitAudioTrack` — a file the app knows whole.
/// - `attachAudioPlayer(_:input:)` → `TrideKitAudioPlayerTap` — what an `AVPlayer` plays, an endless
///   stream such as internet radio included.
/// - `attachAudioEngine(_:node:input:)` → `TrideKitAudioEngineTap` — what a node of the app's own
///   `AVAudioEngine` outputs (its main mixer by default), the app's effects included.
/// - `detachAudio(input:)`, `audioInputs`.
/// - `heardAudio(_:)` (`@_spi(TrideTesting)`) — the audible time and gain an input's last step got.
///
/// ## Non-obvious constraints
/// - No microphone, no permission prompt: TrideKit never records and never changes the app's audio
///   session (its category and activation stay the app's).
/// - The runtime analyses the sound — the same analysis on every platform; TrideKit only hands it
///   the samples and, before every step, the audible time and the gain.
/// - One sound per input (default `music`): attaching to a busy input detaches what was there.

#if canImport(TrideKitFilamentBridge)

import AVFoundation
import Foundation

#if canImport(TrideCore)
import TrideCore
#endif

@available(iOS 17.0, macCatalyst 17.0, *)
extension TrideKitScene {
    /// Play a track and let the scene hear it as `input` (`audio.<input>.<probe>`): a local file the
    /// platform decodes — mp3, AAC, ALAC, WAV, … The scene hears all of it from the start, so seeking
    /// costs nothing. Start it with the track's `play()`.
    ///
    /// - Throws: the decoder's error for a file it cannot read; `TrideError.InvalidAudio` for an
    ///   input name the scene cannot use (lowercase letters, digits and `_`, starting with a letter).
    @discardableResult
    public func attachAudioTrack(url: URL, input: String = "music", loop: Bool = false) throws -> TrideKitAudioTrack {
        TrideKitThreadRule.requireMain()
        let track = try TrideKitAudioTrack(url: url, input: input, loop: loop)
        do {
            try audio.attach(input, track, to: trideScene, report: audioReport())
        } catch {
            track.release()
            throw error
        }
        return track
    }

    /// Let the scene hear what `player` plays as `input` — a file or an endless stream such as
    /// internet radio, followed through the player's pauses, seeks and loops and across its items:
    /// the scene hears the moment the player's clock is at. The app keeps driving the player.
    ///
    /// - Throws: `TrideError.InvalidAudio` for an input name the scene cannot use.
    @discardableResult
    public func attachAudioPlayer(_ player: AVPlayer, input: String = "music") throws -> TrideKitAudioPlayerTap {
        TrideKitThreadRule.requireMain()
        let tap = TrideKitAudioPlayerTap(player: player, input: input)
        do {
            try audio.attach(input, tap, to: trideScene, report: audioReport())
        } catch {
            tap.release()
            throw error
        }
        return tap
    }

    /// Let the scene hear what `node` of the app's `engine` outputs as `input`, as the listener hears
    /// it: the app's own processing before the node (a filter, a fade, the volume) included. The
    /// default node is the one that feeds the engine's output — its main mixer. The app keeps
    /// driving the engine; while it is stopped or paused the scene hears silence.
    ///
    /// - Throws: `TrideError.InvalidAudio` for an input name the scene cannot use, for a node that
    ///   is not attached to `engine`, for the engine's input node (the scene never hears the
    ///   microphone), and for a node whose output is not 32-bit float samples.
    @discardableResult
    public func attachAudioEngine(
        _ engine: AVAudioEngine, node: AVAudioNode? = nil, input: String = "music"
    ) throws -> TrideKitAudioEngineTap {
        TrideKitThreadRule.requireMain()
        let fed = engine.inputConnectionPoint(for: engine.outputNode, inputBus: 0)?.node
        let tap = try TrideKitAudioEngineTap(engine: engine, node: node ?? fed ?? engine.mainMixerNode, input: input)
        do {
            try audio.attach(input, tap, to: trideScene, report: audioReport())
        } catch {
            tap.release()
            throw error
        }
        deliverDiagnostics()
        return tap
    }

    /// The scene stops hearing `input`: a track attached here stops playing, a player or an
    /// engine plays on.
    public func detachAudio(input: String = "music") {
        TrideKitThreadRule.requireMain()
        audio.detach(input, from: trideScene) { note($0) }
        deliverDiagnostics()
    }

    /// The inputs the scene hears now.
    public var audioInputs: [String] {
        TrideKitThreadRule.requireMain()
        return audio.names
    }

    /// The audible time and gain `input` gave the scene's last step (`nil`: not attached or no step
    /// since) — a test compares the runtime's analysis at exactly these.
    @_spi(TrideTesting) public func heardAudio(_ input: String) -> (time: Double, gain: Float)? {
        audio.heard[input]
    }

    /// Problems of a feed or a tap (any thread) reach the scene's diagnostics on the main actor.
    private func audioReport() -> @Sendable (TrideKitDiagnostic) -> Void {
        { [weak self] diagnostic in
            Task { @MainActor in
                self?.note(diagnostic)
                self?.deliverDiagnostics()
            }
        }
    }
}

#endif
