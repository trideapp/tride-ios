/// The SwiftUI view that shows a `TrideKitScene`: one call, the view connects the scene to its renderer.
///
/// ## Public API
/// - `TrideSceneView(scene:renderPolicy:pixelScale:preferredFPS:renderOverrides:handlesTouches:onLoaded:onError:)`.
/// - `TrideSceneView.RenderPolicy` — `.continuous` (every display frame) or `.onDemand` (only frames a
///   scene step changed: a paused or off-screen scene costs no GPU).
///
/// ## Non-obvious constraints
/// - The view connects the scene before the first frame, inside the SwiftUI update; `onLoaded` /
///   `onError` arrive on the next main-actor turn, so the host may change `@State` there. `play()`
///   may be called as soon as the scene exists.
/// - One view, one scene: the view takes the scene's identity, so another scene gets a new view. The
///   same scene in a second view at once moves to that view, and `subscribeDiagnostics` says so
///   (`.runtime`, subject `view`); a view shown again later (a card scrolled back) takes it back.
/// - `handlesTouches` may change at any time; a touch in flight is cancelled on the scene.
/// - `preferredFPS` may change at any time: the same view and scene go on at the new rate.
/// - `renderOverrides` may change at any time: the same view and scene draw with the new knobs from the
///   next frame, the scene's animations untouched.

#if (os(iOS) || targetEnvironment(macCatalyst)) && canImport(SwiftUI) && canImport(MetalKit) && canImport(TrideKitFilamentBridge)

import SwiftUI
import TrideCore

@available(iOS 17.0, macCatalyst 17.0, *)
public struct TrideSceneView: View {
    /// When the view draws a frame.
    public enum RenderPolicy: Sendable {
        /// Every display frame — for scenes with temporal effects (TAA).
        case continuous
        /// Only frames a scene step changed: a paused scene, or one scrolled off screen, draws nothing.
        case onDemand
    }

    private let scene: TrideKitScene
    private let renderPolicy: RenderPolicy
    private let pixelScale: CGFloat?
    private let preferredFPS: Int?
    private let renderOverrides: TrideRenderOverrides?
    private let handlesTouches: Bool
    private let onLoaded: (@MainActor () -> Void)?
    private let onError: (@MainActor (Error) -> Void)?

    /// - Parameters:
    ///   - pixelScale: drawable pixels per point; `nil` = the screen's. Below it (e.g. 2 on a 3× phone)
    ///     a soft scene costs the GPU less.
    ///   - preferredFPS: frames drawn a second; the scene's logic is woken no more often but still runs
    ///     at 60 steps a second, so animations and timings stay the same. `nil` = today's rate (60
    ///     draws); above the screen's rate = the screen's; 0 or less = the default and one diagnostic.
    ///     Change it at any time — e.g. 30 on a hot or slow device.
    ///   - renderOverrides: the host's render knobs over the scene's render settings — MSAA on or off
    ///     (2, 4 or 8 samples), bloom, shadows, depth of field off; every `nil` field keeps the scene's
    ///     value, and `nil` draws the scene exactly as designed. Change it at any time, e.g. MSAA and
    ///     bloom off while `ProcessInfo.thermalState` is `.serious`. Same as
    ///     `TrideKitScene.setRenderOverrides(_:)`.
    ///   - handlesTouches: the view's touches go to the scene's own interactions (taps, drags, and a
    ///     follow of the finger while it touches; on the lift the scene hears the pointer leave);
    ///     `false` leaves them to the host. Either way the scene learns the view's size.
    ///   - onLoaded: the scene is on screen.
    ///   - onError: the view could not load the scene.
    public init(
        scene: TrideKitScene,
        renderPolicy: RenderPolicy = .continuous,
        pixelScale: CGFloat? = nil,
        preferredFPS: Int? = nil,
        renderOverrides: TrideRenderOverrides? = nil,
        handlesTouches: Bool = true,
        onLoaded: (@MainActor () -> Void)? = nil,
        onError: (@MainActor (Error) -> Void)? = nil
    ) {
        self.scene = scene
        self.renderPolicy = renderPolicy
        self.pixelScale = pixelScale
        self.preferredFPS = preferredFPS
        self.renderOverrides = renderOverrides
        self.handlesTouches = handlesTouches
        self.onLoaded = onLoaded
        self.onError = onError
    }

    public var body: some View {
        FilamentMetalView(
            sceneData: scene.sceneGltfData,
            sceneResources: scene.sceneResources,
            renderPolicy: renderPolicy,
            pixelScale: pixelScale,
            preferredFPS: preferredFPS,
            renderOverrides: renderOverrides,
            scene: scene,
            handlesTouches: handlesTouches,
            onLoadError: { [onError] error in
                Task { @MainActor in onError?(error) }
            },
            onLoaded: { [scene, onLoaded] bridge in
                // Inside the SwiftUI update: connect now (before the first frame), tell the host later.
                scene.attachBridge(bridge)
                Task { @MainActor in onLoaded?() }
            }
        )
        .id(ObjectIdentifier(scene))
    }
}

#endif
