/// SwiftUI `MTKViewRepresentable` owning the `FilamentBridge` lifecycle — the renderer under `TrideSceneView`.
///
/// ## Public API (`@_spi(TrideTesting)`: Tride's own test hosts; apps use `TrideSceneView`)
/// - `FilamentMetalView` — creates MTKView, builds the first non-nil `scene` once — at creation or on
///   a later update — from the core's scene record (`TrideKitScene.loadByRecord`), and drives
///   `render`/`dispose` through the coordinator.
/// - `FilamentMetalView.RenderPolicy` (= `TrideSceneView.RenderPolicy`) — `.continuous` (every display frame, the default) or `.onDemand`
///   (only frames with something new: the bridge's `needsRender`, raised by a tick that moved the
///   scene, a resize, the app returning to the foreground). A paused scene draws on demand under
///   either, unless its temporal anti-aliasing is on (`Coordinator.draws`).
/// - `pixelScale` — drawable pixels per point (`nil` = the screen's): a soft scene at 2× on a 3× phone
///   draws 44 % of the pixels, the compositor scales it up.
/// - `preferredFPS` — how many times a second the view draws and wakes `scene`'s tick loop;
///   `nil` = today's 60 draws and a tick every display frame. Changes live, nothing is recreated.
///   Filament hears it with the screen's refresh rate (`RenderRate.filamentPace`): its dynamic
///   resolution aims at the view's real frame time.
/// - `renderOverrides` — the host's render knobs, handed to `scene` (`TrideKitScene.setRenderOverrides`)
///   whenever the view's value changes; `nil` from the start leaves the scene's own knobs alone.
/// - `scene` hears the view's size at every layout, touches or not (the camera the view draws
///   through reaches the scene's ticks with it); `handlesTouches` gives it the view's finger too.
///
/// ## Non-obvious constraints
/// - Rendering stays on the main-thread MetalKit delegate path.
/// - One view loads one scene: another scene needs a new view identity (`.id(…)`).
/// - `.onDemand` suits scenes without temporal effects (no TAA): a paused or off-screen scene then
///   costs no GPU at all. A frame Filament skips keeps `needsRender` up — the next display frame retries.
/// - By the view's distance from the screen (`ScreenDistance`), whatever the policy:
///   only a view showing on screen draws; one within a screen of it keeps its swap chain and renderer,
///   one further (or out of its window) gives them back and its layer keeps the last frame; a memory
///   warning takes them from near views too and shrinks far views' layers. The scene keeps ticking.
/// - The view is opaque unless the scene's environment frame is transparent: then the
///   layer lets what lies under the view show where nothing is drawn (the bridge's swap chain is
///   `CONFIG_TRANSPARENT`, the frame premultiplied), set once the scene is attached in `onLoaded`.
/// - A scene the core hands no record of is not built: `onLoadError` gets `TrideKitSceneNotBuiltError`
///   with the core's words. Without `scene` the view only makes its bridge (a host that drives it).

#if (os(iOS) || targetEnvironment(macCatalyst)) && canImport(SwiftUI) && canImport(MetalKit) && canImport(TrideKitFilamentBridge)

import SwiftUI
import MetalKit
import QuartzCore
import os

import TrideCore
import TrideKitFilamentBridge

private let logger = Logger(subsystem: "studio.tride.kit", category: "FilamentMetalView")

/// SwiftUI view that renders a Tride scene with Filament (`FilamentBridge`) on an `MTKView`, and
/// owns the bridge's load / render / dispose lifecycle. iOS 17+.
@available(iOS 17.0, macCatalyst 17.0, *)
@_spi(TrideTesting) public struct FilamentMetalView: UIViewRepresentable {

    /// The scene failed to load. Called on the main thread, inside the SwiftUI update.
    public let onLoadError: ((Error) -> Void)?

    /// The scene is loaded. Called on the main thread, inside the SwiftUI update, with the bridge
    /// passed explicitly (a `@State` the parent set in `onBridgeCreated` is not visible yet). Attach
    /// at once and defer the rest: `onLoaded: { bridge in scene.attachBridge(bridge); Task { @MainActor
    /// in scene.play() } }`.
    public let onLoaded: ((FilamentBridge) -> Void)?

    /// Every bridge the view creates, before any scene is loaded into it (a host that drives the bridge
    /// itself). To connect a `TrideKitScene`, use `onLoaded`.
    public let onBridgeCreated: ((FilamentBridge) -> Void)?

    public typealias RenderPolicy = TrideSceneView.RenderPolicy

    public let renderPolicy: RenderPolicy

    /// Drawable pixels per point; `nil` = the screen's scale.
    public let pixelScale: CGFloat?

    /// Frames drawn a second, and the most the scene's logic is woken (it still steps 60 Hz over the
    /// elapsed time). `nil` = 60 draws and a wake every display frame; above the screen's rate = the
    /// screen's; 0 or less = the default and one warning. A rate above 60 on a ProMotion iPhone needs
    /// the app's `CADisableMinimumFrameDurationOnPhone`.
    public let preferredFPS: Int?

    /// The host's render knobs over the scene's render settings (`TrideKitScene.setRenderOverrides`):
    /// handed to `scene` whenever this value changes; changes live, nothing is recreated. Needs
    /// `scene` (without one it is ignored, with one warning).
    public let renderOverrides: TrideRenderOverrides?

    /// The scene the view shows, built from the core's record on the first update that brings it
    /// (later changes are ignored — show another scene with a new view identity, `.id(…)`). It hears
    /// the view's size at every layout, whether or not it takes the touches — with it the scene's ticks
    /// send the camera the view draws through (picking, Look at Camera tags). `nil` = a host that
    /// drives the bridge itself.
    public let scene: TrideKitScene?

    /// The view's touches go to `scene` — one finger's down / move / up, then cancel on the lift (a
    /// finger does not hover: the scene hears it leave), for its tap, drag and pointer listeners;
    /// `false` leaves them to the host. Don't stack a SwiftUI gesture over a forwarding view.
    /// It may change at any time (a host that lets the scene have the finger only on one screen):
    /// a touch in flight is cancelled on the scene.
    public let handlesTouches: Bool

    public init(
        renderPolicy: RenderPolicy = .continuous,
        pixelScale: CGFloat? = nil,
        preferredFPS: Int? = nil,
        renderOverrides: TrideRenderOverrides? = nil,
        scene: TrideKitScene? = nil,
        handlesTouches: Bool = true,
        onLoadError: ((Error) -> Void)? = nil,
        onLoaded: ((FilamentBridge) -> Void)? = nil,
        onBridgeCreated: ((FilamentBridge) -> Void)? = nil
    ) {
        self.renderPolicy = renderPolicy
        self.pixelScale = pixelScale
        self.preferredFPS = preferredFPS
        self.renderOverrides = renderOverrides
        self.scene = scene
        self.handlesTouches = handlesTouches
        self.onLoadError = onLoadError
        self.onLoaded = onLoaded
        self.onBridgeCreated = onBridgeCreated
    }

    public final class Coordinator: NSObject, MTKViewDelegate {
        var bridge: FilamentBridge?
        var renderPolicy: RenderPolicy = .continuous
        /// The scene was handed to the bridge (loaded or failed): the bridge takes one scene.
        var sceneLoadAttempted = false
        /// A refused `preferredFPS` was reported (once per view).
        var refusedRateReported = false
        /// The render knobs last handed to the scene; `.none` = none handed yet.
        var handedRenderOverrides: TrideRenderOverrides?? = .none
        /// Knobs without a scene were reported (once per view).
        var knobsWithoutSceneReported = false
        /// Frames the view draws a second (`preferredFramesPerSecond`), for the pace Filament hears.
        var drawRate = 60
        /// The pace last handed to the bridge (`RenderRate.filamentPace`); `nil` = none yet.
        private var handedPace: (refreshRate: Double, interval: Int)?
        /// Display frames the view was asked to draw (instrumentation: the rate tests count them).
        private(set) var drawCallbacks = 0
        /// Where the view stood from the screen at its last display frame.
        private(set) var screenDistance: ScreenDistance = .onScreen
        /// The layer was shrunk to 1 × 1 on a memory warning while far (its last frame went with it).
        private(set) var layerShrunk = false
        /// The bridge drew at least one frame (a view that comes near before it was ever on screen
        /// draws one, so it slides in with a picture).
        private var drewOnce = false
        weak var view: MTKView?
        private var foregroundObserver: NSObjectProtocol?
        private var memoryObserver: NSObjectProtocol?

        override init() {
            super.init()
            // The layer may come back from the background without its last frame.
            foregroundObserver = NotificationCenter.default.addObserver(
                forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main
            ) { [weak self] _ in
                self?.bridge?.setNeedsRender()
            }
            memoryObserver = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.memoryWarning() }
            }
        }

        deinit {
            if let foregroundObserver { NotificationCenter.default.removeObserver(foregroundObserver) }
            if let memoryObserver { NotificationCenter.default.removeObserver(memoryObserver) }
        }

        public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            // A shrunk layer's 1 × 1 is no size of the view: the scene keeps its frame and camera.
            if layerShrunk { return }
            bridge?.resize(size)
        }

        public func draw(in view: MTKView) {
            drawCallbacks += 1
            guard let bridge else { return }
            let distance = Self.distance(of: view)
            screenDistance = distance
            switch distance {
            case .onScreen:
                unshrink(view)
                handPace(view, bridge)
                if Self.draws(renderPolicy, scene: (view as? PointerMTKView)?.scene, bridge) {
                    if bridge.render() { drewOnce = true }
                }
            case .near:
                // Near draws nothing and keeps everything; a page that never showed a frame (or lost
                // it to a shrink) draws one, so it slides in with a picture.
                if layerShrunk || !drewOnce {
                    unshrink(view)
                    handPace(view, bridge)
                    if bridge.render() { drewOnce = true }
                }
            case .far:
                if bridge.hasSurface { bridge.releaseSurface() }
            }
        }

        /// A view on screen draws what changed (`needsRender`); `.continuous` also every display frame
        /// while its scene plays, or while temporal anti-aliasing makes still frames differ. A paused
        /// scene without it would draw the same picture again: it draws on demand. A view without a
        /// scene (a host driving the bridge) keeps every frame.
        @MainActor
        static func draws(_ policy: RenderPolicy, scene: TrideKitScene?, _ bridge: FilamentBridge) -> Bool {
            if bridge.needsRender { return true }
            guard policy == .continuous else { return false }
            guard let scene, !scene.isPlaying else { return true }
            return bridge.temporalAntiAliasing
        }

        /// The draw rate and the screen's refresh rate reach Filament when either changes (a rate
        /// change, the view moving to another screen).
        @MainActor
        private func handPace(_ view: MTKView, _ bridge: FilamentBridge) {
            guard let screen = view.window?.windowScene?.screen else { return }
            let pace = RenderRate.filamentPace(
                drawRate: drawRate, screenMaximum: screen.maximumFramesPerSecond, phoneHeldAt60: Self.phoneHeldAt60)
            if let handedPace, handedPace == pace { return }
            handedPace = pace
            bridge.setFrameRate(displayRefreshRate: pace.refreshRate, frameInterval: pace.interval)
        }

        /// An iPhone app draws at most 60 a second unless its Info.plist opts out.
        @MainActor private static let phoneHeldAt60: Bool = {
            UIDevice.current.userInterfaceIdiom == .phone
                && (Bundle.main.object(forInfoDictionaryKey: "CADisableMinimumFrameDurationOnPhone") as? Bool) != true
        }()

        /// A view out of its window draws nothing: its swap chain and renderer go, the scene stays.
        @MainActor
        func leftWindow() {
            screenDistance = .far
            if let bridge, bridge.hasSurface { bridge.releaseSurface() }
        }

        /// Am. C on a memory warning: near views give their swap chain and renderer back, far ones
        /// also shrink their layer (its three drawables).
        @MainActor
        func memoryWarning() {
            guard let bridge else { return }
            switch screenDistance {
            case .onScreen:
                return
            case .near:
                if bridge.hasSurface { bridge.releaseSurface() }
            case .far:
                if bridge.hasSurface { bridge.releaseSurface() }
                if let view, !layerShrunk {
                    layerShrunk = true
                    view.drawableSize = CGSize(width: 1, height: 1)
                }
            }
        }

        @MainActor
        private func unshrink(_ view: MTKView) {
            guard layerShrunk else { return }
            layerShrunk = false
            let scale = view.contentScaleFactor
            view.drawableSize = CGSize(width: max(1, view.bounds.width * scale), height: max(1, view.bounds.height * scale))
        }

        /// The view's frame in its window against the window's bounds (the screen it shows on).
        @MainActor
        private static func distance(of view: MTKView) -> ScreenDistance {
            guard let window = view.window else { return .far }
            return ScreenDistance.of(frame: view.convert(view.bounds, to: nil), screen: window.bounds)
        }
    }

    public func makeCoordinator() -> Coordinator { Coordinator() }

    public func makeUIView(context: Context) -> MTKView {
        let device = MTLCreateSystemDefaultDevice()
        if device == nil {
            logger.fault("No Metal device available — Filament cannot render")
        }
        let view = PointerMTKView(frame: .zero, device: device)
        view.handlesTouches = handlesTouches
        view.scene = scene
        context.coordinator.renderPolicy = renderPolicy
        view.delegate = context.coordinator
        context.coordinator.view = view
        applyRenderRate(to: view, context.coordinator)
        handRenderOverrides(context.coordinator)
        view.framebufferOnly = false
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.isOpaque = true
        if let pixelScale { view.contentScaleFactor = pixelScale }

        guard let metalLayer = view.layer as? CAMetalLayer else {
            preconditionFailure("MTKView must be backed by CAMetalLayer")
        }
        // The first drawableSize is .zero: start at 1×1, the real size arrives through
        // mtkView(_:drawableSizeWillChange:).
        let initialSize = view.drawableSize == .zero
            ? CGSize(width: 1, height: 1)
            : view.drawableSize

        let bridge = FilamentBridge(metalLayer: metalLayer, size: initialSize)
        context.coordinator.bridge = bridge
        onBridgeCreated?(bridge)

        loadSceneIfNeeded(context.coordinator, view: view)
        return view
    }

    private func loadSceneIfNeeded(_ coordinator: Coordinator, view: MTKView) {
        guard let bridge = coordinator.bridge, let scene, !coordinator.sceneLoadAttempted else { return }
        coordinator.sceneLoadAttempted = true
        do {
            try scene.load(into: bridge)
            onLoaded?(bridge)
            // The host attached the scene in `onLoaded`: its environment says whether the frame has alpha.
            let transparent = bridge.environmentTransparent
            view.isOpaque = !transparent
            view.backgroundColor = transparent ? .clear : nil
        } catch {
            logger.error("scene load failed: \(error.localizedDescription, privacy: .public)")
            onLoadError?(error)
        }
    }

    public func updateUIView(_ uiView: MTKView, context: Context) {
        // The size comes through MTKViewDelegate; the bridge is never recreated — a SwiftUI body update
        // (an overlay changing) must not lose the scene. Only the policy, rate, knobs, scale, scene and
        // touches follow.
        context.coordinator.renderPolicy = renderPolicy
        applyRenderRate(to: uiView, context.coordinator)
        handRenderOverrides(context.coordinator)
        if let pixelScale, uiView.contentScaleFactor != pixelScale { uiView.contentScaleFactor = pixelScale }
        if let view = uiView as? PointerMTKView {
            if view.scene !== scene { view.scene = scene }
            if view.handlesTouches != handlesTouches { view.handlesTouches = handlesTouches }
        }
        loadSceneIfNeeded(context.coordinator, view: uiView)
    }

    /// The draw rate on the view and the wake rate on the scene's tick loop; a refused value
    /// keeps the default and is reported once — on the scene's diagnostics when there is one.
    private func applyRenderRate(to view: MTKView, _ coordinator: Coordinator) {
        let rate = RenderRate.resolve(preferredFPS)
        if rate.refused, let requested = preferredFPS, !coordinator.refusedRateReported {
            coordinator.refusedRateReported = true
            let message = RenderRate.refusalMessage(requested)
            if let scene {
                scene.note(TrideKitDiagnostic(kind: .runtime, subject: "preferredFPS", message: message))
            } else {
                logger.warning("\(message, privacy: .public)")
            }
        }
        // 60 is the draw rate the view always had.
        let fps = rate.fps ?? 60
        if view.preferredFramesPerSecond != fps { view.preferredFramesPerSecond = fps }
        coordinator.drawRate = fps
        scene?.setWakeRate(rate.fps)
    }

    /// The view's render knobs reach the scene when they change (the scene re-applies its settings on
    /// the same bridge). A view that never had knobs leaves the ones the host set on the scene.
    private func handRenderOverrides(_ coordinator: Coordinator) {
        if coordinator.handedRenderOverrides == .none, renderOverrides == nil { return }
        guard coordinator.handedRenderOverrides != .some(renderOverrides) else { return }
        coordinator.handedRenderOverrides = .some(renderOverrides)
        if let scene {
            scene.setRenderOverrides(renderOverrides)
        } else if !coordinator.knobsWithoutSceneReported {
            coordinator.knobsWithoutSceneReported = true
            logger.warning("renderOverrides need the view's scene: without one the knobs are ignored")
        }
    }

    public static func dismantleUIView(_ uiView: MTKView, coordinator: Coordinator) {
        coordinator.bridge?.dispose()
        coordinator.bridge = nil
    }
}


/// `MTKView` that tells its `TrideKitScene` the view's size whenever it lays out and, with
/// `handlesTouches`, hands it one finger in view points (`FingerPhase`): down / move, and on the lift
/// up then cancel — a finger does not hover, so the scene hears it leave. Other fingers are ignored
/// (one pointer); without a scene it behaves as a plain `MTKView`.
@available(iOS 17.0, macCatalyst 17.0, *)
final class PointerMTKView: MTKView {
    /// The scene the view shows — it hears the size with or without the touches.
    weak var scene: TrideKitScene? {
        didSet {
            guard oldValue !== scene else { return }
            cancelTouch(on: oldValue)
            reportSize()
        }
    }

    var handlesTouches = false {
        didSet { if !handlesTouches { cancelTouch(on: scene) } }
    }

    /// The touch being followed.
    private weak var tracked: UITouch?

    private var touchScene: TrideKitScene? { handlesTouches ? scene : nil }

    /// A scene that loses the finger mid-touch hears it cancelled (its drag stops, nothing thrown),
    /// and the next finger is free.
    private func cancelTouch(on lost: TrideKitScene?) {
        guard let touch = tracked else { return }
        tracked = nil
        lost?.pointer(.cancel, at: touch.location(in: self))
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        reportSize()
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window == nil { (delegate as? FilamentMetalView.Coordinator)?.leftWindow() }
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with: event)
        guard let scene = touchScene, tracked == nil, let touch = touches.first else { return }
        tracked = touch
        scene.finger(.began, at: touch.location(in: self))
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesMoved(touches, with: event)
        forward(.moved, touches)
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesEnded(touches, with: event)
        forward(.ended, touches)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesCancelled(touches, with: event)
        forward(.cancelled, touches)
    }

    private func forward(_ phase: FingerPhase, _ touches: Set<UITouch>) {
        guard let scene = touchScene, let touch = tracked, touches.contains(touch) else { return }
        if phase == .ended || phase == .cancelled { tracked = nil }
        scene.finger(phase, at: touch.location(in: self))
    }

    private func reportSize() {
        guard let scene, scene.viewportSize != bounds.size else { return }
        scene.setViewport(size: bounds.size)
    }
}

#endif  // canImport SwiftUI && MetalKit
