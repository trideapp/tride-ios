# TrideKit API

A `.trd` scene is a glTF scene plus its logic: a state machine (layers, states, transitions),
timelines, bindings and events. The logic runs in the Tride runtime at a fixed 60 Hz step; the app does
not repeat it — it only reports facts and listens to what the scene answers.

## `TrideKitScene` — the scene

```swift
let scene = try await TrideKitScene.load(data)   // reads the archive off the main thread
let same = try TrideKitScene(trdBytes: data)     // the same, on the calling thread
// Both throw on a broken file or one from a newer editor.
```

Loading throws `TrideError`:

| Error | When |
|---|---|
| `.LoadFailed` | not a `.trd`; a file that unpacks to more than 1 GiB; a broken manifest or schema; damaged 3D content (an index pointing nowhere, data past the end of its buffer, an index past its mesh's vertices — what the renderer would refuse); the scene's logic does not compile (a formula, the state machine, the bindings graph). The message says which |
| `.NewerFormat` | the scene comes from a newer editor than the TrideKit in the app |

| | |
|---|---|
| `play()` / `pause()` / `isPlaying` | the scene's clock (CADisplayLink); a paused scene stands still and costs no GPU. The scene is `@Observable`: a view reading `isPlaying` updates with it |
| `tickOnce(dtMs:)` | one step by hand — e.g. `tickOnce(dtMs: 0)` while paused, to apply new values |
| `availableBindings()` | what the scene declares — `TrideKitBinding`: `name`, `kind` (`.number`, `.bool`, `.enum`, `.vec3`, `.quat`, `.color`, `.trigger`, `.image`, `.other`), enum `options`, `computed` (the scene computes it: read, never write) |
| `getBinding(_:)` | a binding's current value (`TrideValue`); for an image input, its status as `.string` |
| `validateBindings(_:)` | check that the scene declares the names the app needs (`MissingBindingsError`) |
| `setBinding(_:_:)` | write a value (`throws`): `.number(value:)`, `.bool(value:)`, `.string(value:)` for an enum. `vec3` / `color` / `quat` cannot be written by the host yet — an error |
| `trigger(_:)` | fire a trigger (a one-shot signal, `throws`) |
| `setImage(_:data:)`, `imageLoading(_:)`, `clearImage(_:)`, `imageInput(_:)` | the app's picture in an image input — see "Pictures from the app" (the first three throw for a name that is no image input) |
| `bindInput(_:to:mapping:)` (`throws`), `unbindInput(_:)`, `boundInputs()` | bind a scroll source to one of the scene's number inputs, let go of it, list the bindings — see "The scene follows a scroll view" |
| `scroll(offset:extent:sceneTop:sceneHeight:)`, `dragAxes()` | report the scroll view as it moves; ask which swipes the scene's own drags claim — same section |
| `subscribeEvents(_:)` | the scene's events (`TrideKitEvent`: `name`, `payload: [String: TrideValue]`), on the main thread, in the same frame |
| `subscribeStateChanges(_:)` | state transitions per layer (`TrideKitStateChange`) |
| `subscribeDiagnostics(_:)` | problems the scene runs into (`TrideKitDiagnostic`: `kind` — `evaluation` / `load` / `pointer` / `runtime`, `subject`, `message`), each once, after the tick that met it |
| `currentState(ofLayer:)`, `layerCount`, `layerName(at:)` | where the state machine is now |
| `reset()` | back to how the scene loaded — initial states, default values, the glTF pose — plus one zero-time step, so the initial `onEnter` events arrive again. A trigger named `reset` is an ordinary trigger: `trigger("reset")` |
| `advance(scriptJSON:)` | run a host script (like "Fast-forward" in the editor) and stop at its end: `TrideKitAdvance` — `.completed(ticks:)` or `.stopped(step:reason:ticks:)` (an unreadable script stops at step 0); throws only when the runtime fails |
| `TrideKitScene.buildInfo` | which Tride the app carries (`TrideKitBuildInfo`: `version`, `commit`, `formatVersion`) — no scene needed; attach it to a bug report |

A written value reaches the scene on the next step. A wrong write changes nothing and throws `TrideError`:
an unknown name or a wrong kind — `.InvalidBinding`, an unknown trigger — `.InvalidTrigger`. The names an
app needs are easiest to check once after loading — `validateBindings`.

### Threads

The same rule as TrideKit for Android:

- **Any thread** — `TrideKitScene.load(_:)` and the calls that only talk to the scene's logic: `setBinding`,
  `getBinding`, `trigger`, `setInputs`, `inputsJSON`, `availableBindings`, `validateBindings`, `currentState`, `layerCount`,
  `layerId(at:)`, `layerName(at:)`, `layerCurrentState(at:)`, `currentState(ofLayer:)`, `bindInput`,
  `unbindInput`, `boundInputs`, `dragAxes`. They are `nonisolated`: a network callback or a background task
  writes straight in, and the value lands on the next step. The scene's events still arrive on the main thread.
- **Main thread** — everything that steps, draws, touches, subscribes or attaches: `play`, `pause`, `reset`,
  `tickOnce`, `advance`, `setViewport`, `pointer`, `pick`, `scroll`, `subscribe…`, `unsubscribe`,
  sound, pictures (`imageLoading`, `clearImage`, `imageInput`; `setImage` is `async` and
  switches by itself). They are `@MainActor`: in a Swift 6 app the compiler refuses a call from elsewhere. A
  Swift 5 app can still make one — from a delegate or a completion handler running on another queue — and it
  stops the app at that call with a message naming the call and the rule, instead of a crash in a later
  frame. Switch with `await MainActor.run { … }` or `DispatchQueue.main.async { … }`.

## `TrideSceneView` — the view

A SwiftUI view that draws the scene with Filament (Metal): `TrideSceneView(scene:)`.

| | |
|---|---|
| `renderPolicy` | `.continuous` — every frame; `.onDemand` — only frames a scene step changed (saves battery) |
| `pixelScale` | pixels per point; below the screen's (e.g. 2 on a 3× phone) costs the GPU less |
| `preferredFPS` | frames drawn a second, e.g. `30` on a slow or hot phone; the scene's logic is woken no more often but still runs 60 steps a second (animations and timings stay the same). `nil` (default) — 60; above the screen's rate — the screen's (above 60 on ProMotion needs `CADisableMinimumFrameDurationOnPhone` in the app's Info.plist); 0 or less — the default and one diagnostic. Change it at any time: the scene is not reloaded |
| `renderOverrides` | render switches over the scene's own settings, see "Render knobs"; `nil` (default) — the scene's settings. Change it at any time: the scene is not reloaded |
| `handlesTouches` | the view's touches go to the scene's own interactions (default `true`) |
| `onLoaded` | the scene is on screen |
| `onError` | the view could not load the scene |

The view connects the scene to its renderer itself, before the first frame; `onLoaded` / `onError` arrive
outside the SwiftUI update, so they may change `@State`. `play()` may be called as soon as the scene
exists.

One view shows one scene. To switch scenes, create a new `TrideKitScene` and give it to the view (the
view takes the scene's identity); the old scene stops by itself once nothing references it. A scene given
to a second view while another view still shows it moves to the new view: the earlier view keeps its last
frame, and `subscribeDiagnostics` reports it once (`kind` `.runtime`, subject `view`). Give each view its
own `TrideKitScene` — for a thumbnail and its full-screen view, load the file twice. A view shown again after
the earlier one is gone (a card scrolled back into a list) takes the scene back without a report.

Several views may be on one screen, each with its own scene; with `.onDemand` each draws on its own,
also while the shaders of its first frames are still compiling.

## Render knobs

The scene's render settings are the designer's. The app may change four switches over them, for example
on a hot phone: `TrideRenderOverrides(msaa:msaaSamples:bloom:shadows:depthOfField:)`, every field `nil` by
default. MSAA goes both ways (`msaaSamples` 2, 4 or 8; another count keeps the scene's and reports one
diagnostic); bloom, shadows and depth of field can only be switched off (`true` keeps the scene's). The
scene's other settings (bloom strength, fog, tone mapping) stay as the designer set them, and animated
values go on. Set the knobs on the view (`renderOverrides`) or on the scene
(`scene.setRenderOverrides(_:)`, read back with `scene.renderOverrides`); `nil` returns the scene's
settings, and `scene.sceneRenderSettings` is what the scene itself sets. A change applies live, on the
same scene. The player never changes them on its own:

```swift
let hot: Set<ProcessInfo.ThermalState> = [.serious, .critical]
NotificationCenter.default.addObserver(
    forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main
) { _ in
    let knobs = hot.contains(ProcessInfo.processInfo.thermalState)
        ? TrideRenderOverrides(msaa: false, bloom: false) : nil
    MainActor.assumeIsolated { scene.setRenderOverrides(knobs) }
}
```

## Camera and gestures

The scene sets the camera itself (the active camera from the editor). If the scene reads a number — e.g.
a `turn` binding — the app's gesture just writes it: `setBinding("turn", .number(value: radians))`.

Touches for the scene. If the scene has interactions of its own (in the editor, the Interactions panel:
a tap on a node, a drag that writes a binding), the view hands it its touches:

```swift
TrideSceneView(scene: scene)   // handlesTouches: true by default
```

The view sends the touches to `scene.pointer(_:at:)` and tells the scene its size
(`setViewport(size:)`); the scene finds the node under the finger and runs its actions on the next step.
Without the view, make the same calls by hand: `pointer(.down / .move / .up / .cancel, at: point)` in
view points from the top-left corner; `pick(at:)` — what lies under a point (nearest nodes first),
without changing the scene.

## The scene follows a scroll view

A scene can answer the scroll view it sits in — a hero that speeds up while the page is flung, a plant that
grows as it comes into view — without knowing about scroll views: it declares a plain **number input**
(say `boost`), and the app binds what it measures to it. The scene file holds no scroll interaction.

| `TrideInputSource` | What the input reads | Default range |
|---|---|---|
| `.scrollSpeed` | how fast the scroll view moves, up or down alike, in screen heights per second; back to 0 about a tenth of a second after it stops | 0…2 onto 0…1 |
| `.scrollPosition` | the scene's pass through the scroll view: 0 as its top comes in at the bottom, 1 as its bottom leaves past the top | 0…1 onto 0…1 |

```swift
// Once, after the scene is loaded: the input "boost" now follows the scroll speed.
try scene.bindInput("boost", to: .scrollSpeed)
// A different range: the scene's pass from 20 % to 80 % drives "grow" from 0 to 1.
try scene.bindInput(
    "grow", to: .scrollPosition,
    mapping: TrideInputMapping(fromLow: 0.2, fromHigh: 0.8, toLow: 0, toHigh: 1))

// In your UIScrollViewDelegate, whose scroll view holds the scene's view in its content:
func scrollViewDidScroll(_ scrollView: UIScrollView) {
    let box = sceneView.convert(sceneView.bounds, to: scrollView)   // where the scene sits in the content
    scene.scroll(
        offset: scrollView.contentOffset.y,   // how far it is scrolled
        extent: scrollView.bounds.height,     // its visible height
        sceneTop: box.minY, sceneHeight: box.height)
}
```

| | |
|---|---|
| `bindInput(_:to:mapping:)` | from the next step the scene writes the input from the source, mapped by `TrideInputMapping(fromLow:fromHigh:toLow:toHigh:)` (clamped; `nil` — the default range above). Binding an input again replaces its binding. Throws `TrideError.InvalidBinding` for an input the scene lacks, one that is not a number, one the scene computes or drives by its own pointer interactions, and an empty or non-finite range |
| `unbindInput(_:)` | the app may write the input again; it keeps its value. `true` when it was bound |
| `boundInputs()` | the bindings, in the order they were made: `TrideBoundInput` — `input`, `source`, `mapping` |
| `scroll(offset:extent:sceneTop:sceneHeight:)` | the scroll view in points, all from the start of its content: how far it is scrolled, its visible height, and where the scene's top and its height lie |
| `dragAxes()` | `TrideDragAxes` (`x`, `y`): the directions the scene's own drag interactions claim. `y` means it wants vertical swipes too, which a scroll view around it would otherwise take |

- **A bound input belongs to its source.** `setBinding` on it throws until `unbindInput`. `reset()` keeps
  the bindings; a new scene starts with none, so bind again after loading another scene.
- **Nothing observes your scroll view for you.** Call `scroll` once after the scene is loaded and whenever
  any of its four numbers changes (a scroll, a rotation, a layout) while a scroll source is bound. TrideKit
  refuses a value that is not finite, an `extent` of 0 or less and a negative `sceneHeight` with a
  diagnostic (`kind` `.runtime`, subject `scroll`) and keeps the last report.
- **The speed is measured in time, not in frames.** It is the distance over the last tenth of a second,
  divided by `extent`, so the same scrolling reads the same on a 60 and a 120 Hz screen. After `pause()` and
  `play()` the measure starts afresh: what the scroll view moved meanwhile is not a burst of speed.
- **The values are raw.** The scene's author smooths them with a Spring on the input's data binding.

## Sound the scene hears

A scene can move to the music the app plays: its Sound effectors and Sound fields, and any property it
binds to `audio.<input>.<probe>` (bloom strength, a scale, a colour), follow what the listener hears. The
app hands the sound over and the Tride runtime analyses it — the same analysis on every platform. No
microphone and no permission prompt: TrideKit never records and never changes the app's audio session.

```swift
import AVFoundation
import TrideKit

/// What the scene hears: a track TrideKit plays, the radio the app's own `AVPlayer` plays, or the
/// app's own audio engine.
@MainActor
final class SceneSound {
    let scene: TrideKitScene
    private(set) var track: TrideKitAudioTrack?

    init(scene: TrideKitScene) {
        self.scene = scene
    }

    func playTrack(_ file: URL) throws {
        let track = try scene.attachAudioTrack(url: file, loop: true)   // input "music"
        try track.play()
        track.volume = 0.8   // the scene hears it as loud as the listener does
        self.track = track
    }

    func playRadio(_ radio: AVPlayer) throws {
        try scene.attachAudioPlayer(radio)   // input "music" too: the track stops, the radio is heard
        radio.play()                         // the app keeps driving its player
        track = nil
    }

    func hear(_ engine: AVAudioEngine, after lowPass: AVAudioUnitEQ) throws {
        // What the filter outputs, as the listener hears it; the app keeps driving the engine.
        try scene.attachAudioEngine(engine, node: lowPass)
        track = nil
    }

    func stop() {
        scene.detachAudio(input: "music")    // a track stops; a player plays on, unheard by the scene
    }
}
```

| | |
|---|---|
| `attachAudioTrack(url:input:loop:)` | a local file (anything `AVAudioFile` reads: mp3, AAC, ALAC, WAV); TrideKit plays it, and the scene hears all of it from the start, so seeking costs nothing. Returns a `TrideKitAudioTrack`; throws for a file it cannot read |
| `attachAudioPlayer(_:input:)` | what an `AVPlayer` plays — its current item and every next one, an endless stream such as internet radio included; the app keeps driving the player. Returns a `TrideKitAudioPlayerTap`: `isTapped` says whether the scene hears the current item |
| `attachAudioEngine(_:node:input:)` | what a node of the app's own `AVAudioEngine` outputs — by default the node that feeds the engine's output (its main mixer); the app's players, loops, fades and effects before the node are part of the sound. Returns a `TrideKitAudioEngineTap` (`engine`, `node`, `isTapped`); throws `TrideError.InvalidAudio` for the engine's input node (no microphone) or a node of another engine |
| `detachAudio(input:)` | the scene stops hearing `input`: a track attached there stops playing, a player or an engine plays on |
| `audioInputs` | the inputs the scene hears now |
| `TrideKitAudioTrack` | `play()` (throws when the system refuses to start audio), `pause()`, `seek(to:)` in seconds, `currentTime`, `duration`, `isPlaying`, `loop`, `volume`, `isMuted` |

- **Inputs.** `input` is the name the scene listens to: `audio.music.kick` is probe `kick` of input
  `music`, the default. A scene may listen to several inputs, one sound each: attaching to a busy input
  replaces its sound. A name is lowercase letters, digits and `_`, starting with a letter; another throws
  `TrideError.InvalidAudio`.
- **Nothing to call per frame.** Before each step (the scene's clock, `tickOnce`, `advance`) TrideKit tells
  the scene what sounds now, in time with the speakers: a track's position minus the output latency the
  system reports (a Bluetooth headset included), a player's own clock, the moment each render of an
  engine's node reaches the speaker.
- **An engine.** A paused or stopped engine is silence. TrideKit watches every render of the node without
  taking its tap (the app's own `installTap` on it still works) and follows a configuration change (a new
  route, a new rate) by itself.
- **A player.** The scene follows the `AVPlayer` through its pauses, seeks and loops and onto every next
  item, hearing the moment its clock is at (right after a seek, silence until that sound arrives). When the
  scene goes, the item gets its own audio mix back.
- **Volume.** The scene hears what the listener hears: the volume scales its reaction; a muted, paused or
  ended sound is silence.
- **What a player cannot share.** Protected (FairPlay) items and HTTP Live Streams hand no audio to a tap:
  the scene hears silence, and `subscribeDiagnostics` says so once (subject `audio.<input>`). Files and
  progressive streams (an mp3 or AAC radio URL) are heard.
- **The audio session stays the app's.** A track plays through its own `AVAudioEngine` in the category the
  app set: the default respects the silent switch, `.playback` plays regardless. A call silences the
  track and it resumes when the system says it may; a new output route resumes it where it was.
- **The track belongs to the scene.** Keep it while the scene shows; `detachAudio(input:)` stops it.

## Pictures from the app

A scene can show the app's picture — the user's photo from the library, one the app downloaded — in the
places its author marked. In Tride Editor the author adds an **image input** (a binding of kind `.image`,
e.g. `photo`) and binds it to the picture of an Image Texture; the picture standing there in the editor
is the **placeholder**. The scene shows the placeholder until the app hands a picture over, and again
when the picture fails. The picture is the app's: it is never written into a `.trd`, and it stays through
`reset()`.

```swift
import Foundation
import TrideKit

/// The scene's image input "photo": a picture the app downloads, and what the scene shows now.
@MainActor
final class ScenePhoto {
    let scene: TrideKitScene
    private(set) var status = "empty"

    init(scene: TrideKitScene) {
        self.scene = scene
        // `image.photo` comes on every change of the input: status, revision, width, height, reason.
        scene.subscribeEvents { [weak self] event in
            guard let self, event.name == "image.photo" else { return }
            self.status = self.scene.imageInput("photo")?.status ?? "empty"
        }
    }

    func show(from url: URL) async throws {
        try scene.imageLoading("photo")   // the scene sees 'loading' (a spinner); the picture shown stays
        guard let data = try? await URLSession.shared.data(from: url).0 else {
            try scene.clearImage("photo")  // nothing came: the placeholder
            return
        }
        switch try await scene.setImage("photo", data: data) {
        case .loaded: break                                     // shown from the scene's next step
        case .failed(let reason): print("not shown: \(reason)")  // the placeholder is back
        case .superseded: break                                 // a later call for "photo" came first
        }
    }
}
```

| | |
|---|---|
| `setImage(_:data:) async throws -> TrideImageResult` | JPEG, PNG, WebP or HEIC bytes. The input turns `'loading'` at once, the bytes are decoded off the main thread, then `.loaded` (the scene shows the picture from its next step), `.failed(reason:)` (the input turns `'failed'`, the placeholder shows) or `.superseded` (see below) |
| `setImage(_:cgImage:) async throws -> TrideImageResult` | a picture the app already has. A `CGImage` has no orientation: pass it upright |
| `imageLoading(_:) throws` | a picture is on its way and the app has no bytes yet (a download, the photo picker): the scene sees `'loading'`, the picture shown stays |
| `clearImage(_:) throws` | no picture: the placeholder shows |
| `imageInput(_:) -> TrideImageInput?` | what the input shows now: `status`, `revision`, `width` × `height` (`0 × 0` when no picture shows), `reason` (for `'failed'`); `nil` for a name that is not an image input |

- **Four states.** The scene reads the input's status — `'empty'`, `'loading'`, `'loaded'`, `'failed'` —
  in its state machine (a transition on `bindings.photo == 'loaded'`, say). Every change comes to
  `subscribeEvents` as the event `image.<input>` with `status`, `revision`, `width`, `height` and, for a
  failure, `reason`; a failure also comes to `subscribeDiagnostics` once.
- **The newest call wins.** A decode that a later `setImage` / `imageLoading` / `clearImage` of the same
  input overtook ends `.superseded`: the scene never saw that picture, there is nothing to show.
- **No flash on a replacement.** While the next picture loads, the previous one stays on screen.
- **Fill or fit is the author's.** The Image Texture frames the picture as set in the editor: `stretch`
  (the default), `fill` — the placeholder's proportions, filled whole, the picture cropped around its
  centre — or `fit` — the whole picture, the margins in the Image Texture's edge mode (black, say). The
  app passes a picture of any size and proportions.
- **Size.** TrideKit turns the photo upright by its EXIF orientation and scales it down to 2048 px on the
  long side, the proportions kept; colours are sRGB, alpha straight (not premultiplied).
- **One picture, several places.** An input bound to several fields is decoded and uploaded once.
- **Memory.** A 2048 × 1536 photo holds about 12.6 MB of pixels while it shows, plus its GPU texture; both
  go once the input no longer shows it.
- **Errors.** A name that is not an image input of the scene is the app's mistake: `setImage`,
  `imageLoading` and `clearImage` throw `TrideError.InvalidImage` and nothing changes. A picture that
  cannot be read — bytes that do not decode, an empty download — ends `.failed(reason:)`. The same rule on
  Android and the web.
