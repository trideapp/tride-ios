# Changelog

## v0.19.0 — 2026-11-20

- Tride 0.19.0 writes the same format as 0.18 (0.78), and scenes saved by earlier Tride open unchanged. A scene where the author set a Noise's Compute to Per vertex or First octaves per vertex opens in the 0.18 editor without that look, with a warning; players 0.18 draw it correctly.
- Writing an input, firing a trigger or sending a touch from any thread no longer waits for a frame in progress: the write is checked at once (errors as before), read back at once, and taken by the next frame. The low-level core scene gains sealInputs(), which marks what the next frame takes when frames run on another thread.
- A Curl cloud no longer flickers when a value its paths are built from changes: the older paths stay on screen while the new ones build.
- Noise is computed faster: the core reads four lattice points at once with SIMD, natively and in WebAssembly, and gives the same values bit for bit.
- A view's `preferredFPS` now also sets the frame time Filament's dynamic resolution aims at (33 ms at 30 frames a second), and the screen's refresh rate reaches Filament too. A view with no rate set draws as before.
- A hidden object that nothing else uses (as a cloner's source, for example) costs nothing while it is hidden: its mesh, bends, text, effectors and generators are not computed. Shown again, it looks exactly as if it had never been hidden.
- Every `TrideSceneView` in your app now draws with one shared Filament engine and shares its materials: a second scene view opens without compiling shaders and without a second engine's memory. Nothing to change in your code.
- Scenes kept in your own Swift package get their precompiled materials: add the TrideMaterials plug-in to that package target and load the scene with `TrideKitScene.load(data, bundle: .module)` (or `TrideKitScene(trdBytes:bundle:)`). The plug-in now also finds `.trd` files inside a copied resource folder.
- A paused scene no longer draws the same picture again on every screen frame: under the default continuous policy its view draws only when something changes, as under the on-demand policy. A view with temporal anti-aliasing on still draws every frame.
- A playing scene now prepares its frames off the main thread, on one background queue the whole app shares: scrolling and your own animations no longer wait for a heavy scene. The picture is one frame behind a touch (17 ms at 60 frames a second); events still arrive on the main thread, in the frame that shows what they belong to. For this release, `TrideKitScene.defaultTickMode = .mainThread` brings back the old behaviour.
- A scene view only draws while it shows on screen. One scrolled more than a screen away gives back its frame buffers and keeps showing its last frame; on a memory warning views off screen give back more. The scene keeps running, and a view scrolled back draws at once.
- An object whose material makes it fully transparent at that moment (faded to zero opacity, for example) is not drawn and costs no GPU time. A touch still picks it as before, and its children are drawn.
