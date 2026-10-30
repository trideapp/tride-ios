# TrideKit for iOS — Tride 0.13.0

Plays Tride scenes (`.trd`) in an iOS app: the Tride runtime, the Filament renderer and a SwiftUI view on
top of them. A scene is built in Tride Editor ([tride-editor](https://github.com/trideapp/tride-editor))
and exported as `.trd`; the app only writes values into it (bindings) and fires triggers — the scene
decides what to show.

This repository is generated from the Tride source repository: one commit per release, changes are not
accepted here; questions and bugs go to Issues. Licence — `LICENSE` (Tride Players Licence 1.1); third-party
components and their licences — `THIRD_PARTY_NOTICES.md`; which build this release is — `SNAPSHOT.md`;
what changed — `CHANGELOG.md`.

## What's inside

| | |
|---|---|
| `Package.swift` | The Swift package: products `TrideKit` and `TrideCore`. The Tride runtime, Filament and the bridge to it are prebuilt xcframeworks, attached to the release |
| `Sources/` | TrideKit (Swift) and the Swift wrapper of the runtime |
| `Examples/TridePlayer` | An example player: opens any `.trd`, a control for each of its bindings |
| `Docs/` | [API](Docs/API.md) |

## Access

The repository is private. Xcode clones it through your GitHub account and downloads the package's
binaries (release assets) through the GitHub API with a token from `~/.netrc`:

1. Xcode → Settings → Accounts → "+" → GitHub.
2. A token: github.com → Settings → Developer settings → Fine-grained tokens → Generate new token.
   Resource owner — `trideapp`, Repository access — only `tride-ios`, Permissions → Contents: Read-only.
3. `~/.netrc` (and `chmod 600 ~/.netrc`):

```
machine api.github.com
  login <your GitHub login>
  password <token>
```

## In your app

1. File → Add Package Dependencies… → `https://github.com/trideapp/tride-ios` → Dependency Rule
   "Up to Next Major Version" from 0.13.0.
2. Add the **TrideKit** product to your app target (it also exposes the runtime's types — `TrideValue`
   and others; a separate **TrideCore** is only for a target that needs no rendering).
3. In the target's Build Settings: Excluded Architectures → "+" for Any iOS Simulator SDK → `x86_64`
   (`EXCLUDED_ARCHS[sdk=iphonesimulator*] = x86_64`). The simulator libraries are arm64 only: without
   this, a build that is not limited to the active architecture (Any iOS Simulator, Release, CI) fails
   with `ld: … required architecture 'x86_64'`.
4. In the target's Build Phases → Run Build Tool Plug-ins → "+" → **TrideMaterials** (trust it when Xcode
   asks). At each build it turns the materials of the target's `.trd` files into Metal libraries for the
   iPhone; without it, the first run after an install compiles them on the phone (seconds of black screen
   and stutter). The first build takes about 20 s per scene, later builds skip what is already built. A
   `.trd` the app downloads at run time is compiled on the phone, as before.
5. Add a `.trd` to the app and show it:

```swift
import SwiftUI
import TrideKit

struct SceneView: View {
    @State private var scene: TrideKitScene?
    @State private var loadError: String?

    var body: some View {
        ZStack {
            if let scene {
                TrideSceneView(
                    scene: scene,
                    renderPolicy: .onDemand,
                    onError: { loadError = $0.localizedDescription }
                )
                .ignoresSafeArea()
            }
            if let loadError { Text(loadError) }
        }
        .task {
            guard let url = Bundle.main.url(forResource: "scene", withExtension: "trd") else {
                loadError = "scene.trd is not in the app bundle"
                return
            }
            do {
                let scene = try await TrideKitScene.load(Data(contentsOf: url))  // reads it off the main thread
                scene.play()
                self.scene = scene
            } catch {
                loadError = error.localizedDescription  // a broken file or a scene from a newer editor
            }
        }
    }
}
```

`TrideSceneView` connects the scene to its renderer itself; `onLoaded` / `onError` arrive outside the
SwiftUI update, so they may change `@State`. `play()` may be called as soon as the scene exists.

Next: `try scene.setBinding("work_elapsed", .number(value: 42))`, `try scene.trigger("begin")`,
`scene.subscribeEvents { print($0.name) }` — see [Docs/API.md](Docs/API.md). Which Tride an app carries —
`TrideKitScene.buildInfo` (release, commit, `.trd` format); attach it to a bug report.

## Sound the scene hears

A scene can move to the music your app plays: its Sound effectors and Sound fields, and any property it
binds to `audio.<input>.<probe>` (bloom strength, a scale, a colour), follow the sound. No microphone, no
permission. `attachAudioTrack(url:input:loop:)` plays a file the scene hears; `attachAudioPlayer(_:input:)` lets
it hear your `AVPlayer`, internet radio included; `attachAudioEngine(_:node:input:)` lets it hear a node of your
own `AVAudioEngine`, your effects included; `detachAudio(input:)` stops. See [Docs/API.md](Docs/API.md).

```swift
let track = try scene.attachAudioTrack(url: musicURL)   // input "music": TrideKit plays the file
try track.play()
try scene.attachAudioPlayer(radio, input: "radio")      // your AVPlayer; the app keeps driving it
```

## Pictures from the app

A scene can show your app's picture — the user's photo from the library, one the app downloaded — in the
places its author marked. In Tride Editor the author adds an **image input** (a binding of kind `image`,
e.g. `photo`) and binds it to the picture of an Image Texture; the picture already standing there is the
**placeholder**. The scene shows the placeholder until the app hands a picture over, and again when the
picture fails. The picture is the app's: it is never written into a `.trd`.

```swift
import PhotosUI
import SwiftUI
import TrideKit

/// The user's photo from the library in the scene's image input "photo".
struct ScenePhotoPicker: View {
    let scene: TrideKitScene
    @State private var item: PhotosPickerItem?

    var body: some View {
        PhotosPicker("Choose a photo", selection: $item, matching: .images)
            .onChange(of: item) { _, item in
                guard let item else { return }
                Task { @MainActor in
                    do {
                        try scene.imageLoading("photo")   // the scene sees 'loading'; the picture shown stays
                        let data = try? await item.loadTransferable(type: Data.self)
                        switch try await scene.setImage("photo", data: data ?? Data()) {
                        case .loaded: break                       // shown from the scene's next step
                        case .failed(let reason): print(reason)   // the placeholder is back
                        case .superseded: break                   // a later call for "photo" came first
                        }
                    } catch {
                        print(error)   // "photo" is no image input of this scene: the app's mistake, nothing changed
                    }
                }
            }
    }
}
```

- `setImage(_:data:)` takes JPEG, PNG, WebP or HEIC bytes; `setImage(_:cgImage:)` a picture you already
  have (upright: a `CGImage` has no orientation). Both end in `.loaded`, `.failed(reason:)` or
  `.superseded` — a later `setImage` / `imageLoading` / `clearImage` of the same input came first, so the
  scene never saw this picture.
- `imageLoading(_:)` says a picture is on its way before you have its bytes; `clearImage(_:)` takes the
  picture away (the placeholder shows); `imageInput(_:)` reads what the input shows now.
- A name that is no image input of the scene is the app's mistake: `setImage`, `imageLoading` and
  `clearImage` throw `TrideError.InvalidImage` (call them with `try`) and change nothing. A picture that
  cannot be read is not a throw: `setImage` ends `.failed(reason:)`.
- The scene's state machine reads the input's status — `empty`, `loading`, `loaded`, `failed` — and every
  change comes to `subscribeEvents` as the event `image.<input>` (`status`, `revision`, `width`, `height`,
  `reason`).
- Fill or fit is the author's: the Image Texture stretches the picture, fills the placeholder's
  proportions (cropped around the centre) or fits it whole with margins. The app passes any size.
- TrideKit decodes off the main thread, turns the photo by its EXIF orientation and scales it down to
  2048 px on the long side. While a new picture loads the previous one stays: no flash of the placeholder.

More in [Docs/API.md](Docs/API.md).

## The scene follows a scroll view

A scene can answer the scroll view it sits in — it declares a plain number input (say `boost`) and the app
binds what it measures to it: `.scrollSpeed` (how fast the scroll view moves, up or down alike, in screen
heights per second) or `.scrollPosition` (the scene's pass through the view, 0 as its top comes in at the
bottom, 1 as its bottom leaves past the top). Nothing watches your scroll view for you: report it with
`scroll(offset:extent:sceneTop:sceneHeight:)` once after loading and whenever it moves.

```swift
try scene.bindInput("boost", to: .scrollSpeed)   // after the scene is loaded; unbindInput("boost") lets go

// In your UIScrollViewDelegate, whose scroll view holds the scene's view in its content:
func scrollViewDidScroll(_ scrollView: UIScrollView) {
    let box = sceneView.convert(sceneView.bounds, to: scrollView)
    scene.scroll(offset: scrollView.contentOffset.y, extent: scrollView.bounds.height,
                 sceneTop: box.minY, sceneHeight: box.height)
}
```

`bindInput(_:to:mapping:)` takes a `TrideInputMapping` to pick the range, `boundInputs()` lists the
bindings, `dragAxes()` says whether the scene's drags also claim vertical swipes. A bound input belongs to
its source: `setBinding` on it throws until `unbindInput`. See [Docs/API.md](Docs/API.md).

## Example

`Examples/TridePlayer/TridePlayer.xcodeproj` → Run. It takes the package from this clone and the binaries
from the same links (needs `~/.netrc`). For an iPhone, pick your Team in Signing & Capabilities. "Open .trd…"
at the top opens a scene exported from Tride Editor (a file from Files); "Bindings" lists every value and
trigger of the open scene.

## Requirements and versions

- Xcode 26 (tested on 26.3; older versions not tested), a Mac with Apple silicon; iOS 17+, arm64 only:
  iPhone and the simulator on Apple silicon.
- macOS: only the runtime works on a Mac — `import TrideKit` (or `TrideCore`) gives a scene without a
  picture; Apple silicon (arm64) only: an Intel or universal ("Any Mac") build does not link.
- Other Linker Flags: no `-all_load`, and no `-force_load` of another static Rust library. The Tride
  runtime is a static Rust library with its own copy of the Rust standard library; forcing it into an
  app together with another such library fails on duplicate symbols. Plain linking and `-ObjC` are fine.
- A scene opens in TrideKit of the same release or newer; an older TrideKit does not open a scene from a
  newer editor ("newer format"). Take the editor of the same release number from `tride-editor`.
