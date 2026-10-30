/// The audio thread's side of a tap and the timing of what it copied — one machinery for the app's
/// own engine (`TrideKitAudioEngineTap`) and its player (`TrideKitAudioPlayerTap`).
///
/// ## Internal API
/// - `AudioTapFormat` — the 32-bit float PCM a tap sees; the channels fed to the runtime (1 or 2).
/// - `AudioTapRing<Stamp>` — `capture` (the audio thread) / `drain` (the step): the samples of every
///   render with the stamp the source gave it (a host time, an item time).
/// - `AudioTapTimeline` — renders on the input's audio time → the runtime's stream chunks; `head`
///   is the audio time of the next sample.
///
/// ## Non-obvious constraints
/// - One writer (the render) and one reader (the step): the lock guards only the counters and the
///   stamps; the reader copies the samples outside it and drops what a render overwrote meanwhile.
///   Nothing on the audio thread allocates or waits for the copy.
/// - A ring keeps `keepSeconds` for a scene that does not step: the oldest samples go first.
/// - The timeline counts samples and moves the count only where a render's time leaves it: up to
///   `slack` later still continues it (coarse stamps); earlier drops the overlap, so the stream never
///   steps back — unless the source's time went back past `jumpBack` (a player's seek), where it
///   starts over.

import AudioToolbox
import AVFoundation
import Foundation
import os

/// The PCM a tap copies (32-bit float).
struct AudioTapFormat: Equatable {
    let sampleRate: Double
    /// Channels in the source's buffers.
    let sourceChannels: Int
    /// Channels fed to the runtime: 1 or 2.
    let channels: Int
    let interleaved: Bool

    init?(_ format: AVAudioFormat) {
        guard format.commonFormat == .pcmFormatFloat32, format.sampleRate > 0, format.channelCount > 0 else {
            return nil
        }
        sampleRate = format.sampleRate
        sourceChannels = Int(format.channelCount)
        channels = min(2, sourceChannels)
        interleaved = format.isInterleaved
    }

    init?(_ asbd: AudioStreamBasicDescription) {
        guard asbd.mFormatID == kAudioFormatLinearPCM, asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              asbd.mBitsPerChannel == 32, asbd.mSampleRate > 0, asbd.mChannelsPerFrame > 0
        else { return nil }
        sampleRate = asbd.mSampleRate
        sourceChannels = Int(asbd.mChannelsPerFrame)
        channels = min(2, sourceChannels)
        interleaved = asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved == 0
    }
}

/// Renders kept between two drains (a render is ≥ 64 frames: ≥ 2 s of renders).
private let markRoom = 2048

/// A ring of the PCM a source rendered and the stamp of each render, waiting for the scene's step.
final class AudioTapRing<Stamp: Sendable>: @unchecked Sendable {
    /// The renders since the last drain: `pcm` (interleaved `format.channels`) and where each render
    /// starts in it with its stamp (`offset` < 0: the render began before the oldest sample kept).
    struct Drained {
        var pcm: [Float] = []
        var renders: [(offset: Int, stamp: Stamp?)] = []
    }

    private struct Mark {
        var frame = 0
        var stamp: Stamp?
    }

    private struct Counters {
        /// Frames written / handed over since the ring started.
        var written = 0
        var drained = 0
        var marks: [Mark] = []
        var marksWritten = 0
        var marksDrained = 0
        var closed = false
    }

    let format: AudioTapFormat
    private let lock = OSAllocatedUnfairLock(initialState: Counters())
    /// `capacity` frames of `format.channels` interleaved samples: written by the render outside the
    /// lock, published by `written`.
    private let samples: UnsafeMutablePointer<Float>
    private let capacity: Int

    init(format: AudioTapFormat, keepSeconds: Double) {
        self.format = format
        capacity = max(1, Int(keepSeconds * format.sampleRate))
        samples = .allocate(capacity: capacity * format.channels)
        samples.initialize(repeating: 0, count: capacity * format.channels)
        let marks = [Mark](repeating: Mark(), count: markRoom)
        lock.withLock { $0.marks = marks }
    }

    deinit {
        samples.deallocate()
    }

    /// Nothing is copied from now on (any thread).
    func close() {
        lock.withLock { $0.closed = true }
    }

    /// The audio thread: a render of `frames` frames of `list`, stamped `stamp`.
    func capture(_ list: UnsafeMutablePointer<AudioBufferList>, frames: Int, stamp: Stamp?, silent: Bool) {
        guard frames > 0 else { return }
        let buffers = UnsafeMutableAudioBufferListPointer(list)
        let format = format
        let (start, closed) = lock.withLock { ($0.written, $0.closed) }
        guard !closed else { return }
        // Only this thread writes: the slots past `written` are the reader's no more.
        for frame in 0..<frames {
            let slot = ((start + frame) % capacity) * format.channels
            for c in 0..<format.channels {
                samples[slot + c] = silent ? 0 : Self.sample(buffers, format: format, frame: frame, channel: c)
            }
        }
        lock.withLock { r in
            r.marks[r.marksWritten % markRoom] = Mark(frame: start, stamp: stamp)
            r.marksWritten += 1
            r.written = start + frames
        }
    }

    /// One sample of `buffers` (0 where the list is shorter than the format says).
    private static func sample(
        _ buffers: UnsafeMutableAudioBufferListPointer, format: AudioTapFormat, frame: Int, channel: Int
    ) -> Float {
        let source = min(channel, format.sourceChannels - 1)
        if format.interleaved {
            guard let data = buffers.first?.mData?.assumingMemoryBound(to: Float.self),
                  (frame * format.sourceChannels + source + 1) * MemoryLayout<Float>.size <= Int(buffers[0].mDataByteSize)
            else { return 0 }
            return data[frame * format.sourceChannels + source]
        }
        let lane = min(source, buffers.count - 1)
        guard lane >= 0, let data = buffers[lane].mData?.assumingMemoryBound(to: Float.self),
              (frame + 1) * MemoryLayout<Float>.size <= Int(buffers[lane].mDataByteSize)
        else { return 0 }
        return data[frame]
    }

    /// The step: everything rendered since the last drain that is still kept.
    func drain() -> Drained {
        let capacity = capacity
        // The counters and the stamps under the lock; the samples after it.
        let (first, written, marks) = lock.withLock { r -> (Int, Int, [Mark]) in
            let first = max(r.drained, r.written - capacity)
            var marks: [Mark] = []
            // The renders still kept; the one `first` falls in may start before it.
            let oldest = max(r.marksDrained, r.marksWritten - markRoom)
            if first < r.written, oldest < r.marksWritten {
                for k in oldest..<r.marksWritten {
                    let mark = r.marks[k % markRoom]
                    let next = k + 1 < r.marksWritten ? r.marks[(k + 1) % markRoom].frame : r.written
                    if next > first { marks.append(mark) }
                }
            }
            let written = r.written
            r.drained = written
            r.marksDrained = r.marksWritten
            return (first, written, marks)
        }
        guard first < written else { return Drained() }
        let channels = format.channels
        var pcm = [Float](repeating: 0, count: (written - first) * channels)
        pcm.withUnsafeMutableBufferPointer { out in
            guard let base = out.baseAddress else { return }
            // At most two runs: up to the ring's end, then from its start.
            var frame = first
            while frame < written {
                let slot = frame % capacity
                let run = min(written - frame, capacity - slot)
                (base + (frame - first) * channels).update(from: samples + slot * channels, count: run * channels)
                frame += run
            }
        }
        // Frames a render overwrote while they were copied are not theirs any more.
        let lost = max(0, lock.withLock { $0.written } - capacity - first)
        var out = Drained()
        out.pcm = lost > 0 ? Array(pcm.dropFirst(lost * channels)) : pcm
        out.renders = marks.map { ($0.frame - first - lost, $0.stamp) }
        return out
    }
}

/// Renders → the runtime's stream: chunks that continue each other on the input's audio time.
struct AudioTapTimeline {
    /// A render up to this much later than the count continues it.
    let slack: Double
    /// A render this much earlier than the count starts over there; `nil`: the source's time never
    /// goes back, the overlap is dropped.
    let jumpBack: Double?
    /// The audio time of the next sample (`nil`: none yet, or a new start).
    private(set) var head: Double?

    /// A render within this of the count continues it (the runtime continues within 1 ms).
    static let tolerance = 0.001

    init(slack: Double = tolerance, jumpBack: Double? = nil) {
        self.slack = max(Self.tolerance, slack)
        self.jumpBack = jumpBack
    }

    /// The next sample starts a new count.
    mutating func reset() {
        head = nil
    }

    /// `pcm` (interleaved `channels` at `rate`) whose renders start at `renders[i].offset`, heard
    /// from `renders[i].time` (`nil`: it follows the render before). `untimed`: a render without a
    /// time had nothing to follow and was left out.
    mutating func chunks(
        _ pcm: [Float], channels: Int, rate: Double, renders: [(offset: Int, time: Double?)]
    ) -> (chunks: [(start: Double, pcm: [Float])], untimed: Bool) {
        let frames = pcm.count / max(1, channels)
        var out: [(start: Double, pcm: [Float])] = []
        var untimed = false
        var chunk: (start: Double, from: Int)?
        var to = 0
        func flush() {
            guard let open = chunk, to > open.from else { return }
            out.append((open.start, Array(pcm[(open.from * channels)..<(to * channels)])))
            chunk = nil
        }
        for (index, render) in renders.enumerated() {
            var from = max(0, render.offset)
            let end = min(frames, index + 1 < renders.count ? renders[index + 1].offset : frames)
            guard end > from else { continue }
            var start: Double
            if let time = render.time {
                start = time + Double(from - render.offset) / rate
            } else if let head {
                start = head
            } else {
                untimed = true
                continue
            }
            if let head {
                let off = start - head
                if let jumpBack, off < -jumpBack {
                    // The source went back (a seek): the stream starts over there.
                } else if off < -Self.tolerance {
                    // Heard earlier than counted: drop the overlap, never step back.
                    from += Int((-off * rate).rounded())
                    guard from < end else { continue }
                    start = head
                } else if off <= slack {
                    start = head
                }
            }
            // One chunk while the renders follow each other.
            if chunk == nil || start != head || to != from {
                flush()
                chunk = (start, from)
            }
            to = end
            head = start + Double(end - from) / rate
        }
        flush()
        return (out, untimed)
    }
}
