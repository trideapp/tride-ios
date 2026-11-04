/// Orchestrates uniffi `TrideScene` ticks, pointer dispatch, and optional Filament attachment for iOS hosts.
///
/// ## Public API
/// - `TrideKitScene` — owns Rust scene handle, `PointerDispatcher`, `TickDriver`, and weak `FilamentBridge` hookup.
/// - `setViewCameraOrbit(yaw:pitch:distanceScale:)` — host-driven orbit of the default viewport camera.
/// - `subscribeEvents` / `subscribeStateChanges` / `unsubscribe` + `layerCount`, `layerId(at:)`,
///   `layerName(at:)`, `layerCurrentState(at:)`, `currentState(ofLayer:)` — what the scene tells
///   the host: `TrideKitEvent`, `TrideKitStateChange`, `TrideKitSubscription`.
/// - `inputsJSON()` / `setInputs(json:strict:)` — the scene's inputs for a settings screen (every input with
///   its kind, limits and labels; kind `.string` is text) and a checked batch write: the report
///   (`TrideKitInputsReport`) names what was applied, refused with a code, unknown or untouched.
/// - `advance(scriptJSON:)` — replay a host script in one call and show its final state: open a
///   scene "at minute 25" of its timer.
/// - Sound the scene hears (`TrideKitScene+Audio`): `attachAudioTrack(url:input:loop:)`,
///   `attachAudioPlayer(_:input:)`, `attachAudioEngine(_:node:input:)`, `detachAudio(input:)`,
///   `audioInputs`.
/// - The app's pictures for image inputs (`TrideKitScene+Images`): `setImage(_:data:)`,
///   `setImage(_:cgImage:)`, `imageLoading(_:)`, `clearImage(_:)` (each throws for a name that is
///   no image input), `imageInput(_:)`; status changes arrive as `image.<input>` events through
///   `subscribeEvents`.
/// - `setViewport(size:)` / `pointer(_:at:)` / `pick(at:)` — touches for the scene's listeners
///   the core picks and runs taps and drags; the view (`TrideSceneView`,
///   `FilamentMetalView(scene:handlesTouches:)`) reports its size here always, its touches on request.
/// - `bindInput(_:to:mapping:)` / `unbindInput(_:)` / `boundInputs()` — the app binds a source it
///   measures (`TrideInputSource`: its scroll view's speed or the scene's pass through it) to one
///   of the scene's number inputs; the scene never names the source.
/// - `scroll(offset:extent:sceneTop:sceneHeight:)` / `dragAxes()` — the app's scroll view, for the
///   inputs bound to a scroll source; the app calls it, nothing observes a view yet.
/// - `setRenderOverrides(_:)` / `renderOverrides` / `sceneRenderSettings` — the host's render knobs
///   (`TrideRenderOverrides`: MSAA both ways; bloom, shadows, depth of field off) over the scene's
///   render settings, the knobs set, and the settings the scene's designer chose. Cleared over a
///   scene without render settings, the view draws as a new one (`restoreInitialRenderSettings`).
/// - `moduleInstanceCounts()` — per generator, the clones the core evaluated and the bridge draws
///   (test / debug introspection, `TrideModuleInstanceCount`); `renderSettingsDictionary()` — the
///   render settings as the bridge takes them, knobs applied (test hosts try variants).
///
/// ## Non-obvious constraints
/// - Threads (`ThreadRule.swift`): calls that step, draw, touch, subscribe or attach are `@MainActor` and
///   stop a Swift 5 caller off the main thread AT the call; the runtime-only calls (`setBinding`,
///   `trigger`, `setInputs`, `getBinding`, `availableBindings`, `currentState`, layers, `bindInput`, …) are
///   `nonisolated`.
/// - Ticks run even when Metal bridge is nil so headless smoke tests still advance SM state.
/// - Scene signals reach handlers on the main actor right AFTER the tick's mutations went to the
///   bridge (same frame), in the order the runtime produced them; `currentState()` stays the LAST
///   layer's state. Subscribe before `attachBridge` / `play()` to see the initial `onEnter` events.
/// - Ticks are changed-only: a value holds until the core writes it again. Ticks before
///   `attachBridge` reach no renderer, so attaching asks the core for a full frame first. After
///   `reset()` the core's frame is whole: an animated render setting it does not carry goes back
///   to the scene's own value.
/// - A re-evaluated trunk shows one tick later (PointerDispatcher); `pause()`, a paused
///   `tickOnce` and `advance` show it at once.
/// - Before every step (`tickOnce`, the tick loop, `advance`) each attached sound tells the runtime
///   what sounds now.
/// - A step after a move of the bridge's default camera sends it with `set_viewport` first (Look at Camera tags turn toward it on that step).
/// - The showing view's render rate (`preferredFPS`) paces the tick loop's wakes; the core
///   still steps 60 Hz over the elapsed time.
/// - The render knobs lay over the scene's settings in the core, never here; no knob = the scene's
///   settings exactly. A change re-applies them on the same bridge and hands the core's whole frame
///   out again, so an animated bloom strength, saturation, fog colour or pixelation stays where the
///   scene's animation has it; an on-demand view draws one frame.
/// - One view at a time: attaching while another live view shows the scene moves it to the new view
///   and reports a `.runtime` diagnostic (subject `view`); the earlier view keeps its last frame.
/// - A view builds the scene from the core's scene record (`load(into:)`), never from the file's glTF;
///   a scene the core hands no record of is not built (`TrideKitSceneNotBuiltError`). What the scene
///   draws only in part (a `requires` iOS does not draw, a material without values the record lacks)
///   is said once (`partialDrawNotice`, diagnostic `partialDraw`).
/// - The core keeps no glTF carrier of an older file past the load: a view built at any time draws
///   the same meshes from the core's record and datablocks.

import Foundation
import Observation
import os
import simd

#if canImport(TrideCore)
@_exported import TrideCore
#endif

#if canImport(TrideKitFilamentBridge)
import TrideKitFilamentBridge

private let logger = Logger(subsystem: "studio.tride.kit", category: "TrideKitScene")
private let evalLogger = Logger(subsystem: "studio.tride.kit", category: "TrideEval")
/// Signpost intervals for Instruments (`tick`, `dispatch`): free unless a trace records them.
private let signposts = OSSignposter(subsystem: "studio.tride.kit", category: "Performance")

// MARK: - Scene signals

/// Clones of one generator: evaluated by the core, drawn by the bridge (`moduleInstanceCounts()`).
@_spi(TrideTesting) public struct TrideModuleInstanceCount: Equatable, Sendable {
    public let generator: String
    public let evaluated: Int
    public let drawn: Int
}

/// An event the scene reported: a state's `onEnter` / `onExit`, a transition's or a listener's
/// `reportEvent`.
public struct TrideKitEvent: Sendable, Equatable, CustomStringConvertible {
    /// The event's public name (`events[].name`).
    public let name: String
    /// The reporting action's constant payload.
    public let payload: [String: TrideValue]

    public init(name: String, payload: [String: TrideValue] = [:]) {
        self.name = name
        self.payload = payload
    }

    public var description: String {
        guard !payload.isEmpty else { return name }
        let fields = payload.keys.sorted().map { "\($0): \(payload[$0].map(String.init(describing:)) ?? "")" }
        return "\(name) {\(fields.joined(separator: ", "))}"
    }
}

/// One state-machine layer's transition.
public struct TrideKitStateChange: Sendable, Equatable, CustomStringConvertible {
    /// Layer index in execution order — `currentState()` reads the last one.
    public let layerIndex: Int
    /// Layer slug (`layers[].id`).
    public let layerId: String
    /// State the layer left.
    public let from: String
    /// State the layer entered.
    public let to: String
    /// Transition slug — for an edge into a rule node, the rule's id.
    public let transition: String

    public init(layerIndex: Int, layerId: String, from: String, to: String, transition: String) {
        self.layerIndex = layerIndex
        self.layerId = layerId
        self.from = from
        self.to = to
        self.transition = transition
    }

    public var description: String { "\(layerId): \(from) → \(to) (\(transition))" }
}

/// Token of one `subscribeEvents` / `subscribeStateChanges` registration — pass it to
/// `TrideKitScene.unsubscribe(_:)`.
public struct TrideKitSubscription: Hashable, Sendable {
    fileprivate let id: UInt64
}

/// Takes the runtime's callbacks during `TrideScene.tick` — they fire synchronously on the ticking
/// thread, BEFORE the tick's mutations reach the bridge — and keeps them, in order, until
/// `TrideKitScene` hands them to the host after dispatch.
private final class SceneSignalInbox: TrideOutgoingEventListener, TrideLayerStateChangeListener,
    @unchecked Sendable
{
    enum Signal {
        case event(TrideKitEvent)
        case stateChange(TrideKitStateChange)
    }

    private let lock = NSLock()
    private var pending: [Signal] = []

    func onOutgoingEvent(name: String, payload: [String: TrideValue]) {
        let event = TrideKitEvent(name: name, payload: payload)
        lock.withLock { pending.append(.event(event)) }
    }

    func onLayerStateChange(change: TrideStateChange) {
        let stateChange = TrideKitStateChange(
            layerIndex: Int(change.layerIndex),
            layerId: change.layerId,
            from: change.from,
            to: change.to,
            transition: change.transition
        )
        lock.withLock { pending.append(.stateChange(stateChange)) }
    }

    func drain() -> [Signal] {
        lock.withLock {
            defer { pending.removeAll() }
            return pending
        }
    }
}

/// A Tride scene for an iOS host: the Tride runtime (`TrideScene`) plus its connection to the
/// Filament view. Drives a CADisplayLink tick loop and applies each tick's changes to the view.
/// One scene per view: a `FilamentBridge` builds one scene for its lifetime.
///
/// ### Typical usage in SwiftUI
///
/// ```swift
/// @State private var scene: TrideKitScene?
///
/// FilamentMetalView(
///   scene: scene,
///   onLoaded: { bridge in
///       scene?.attachBridge(bridge)
///       Task { @MainActor in scene?.play() }
///   }
/// )
/// .task {
///   scene = try? TrideKitScene(trdBytes: trd)
/// }
/// ```
///
/// `onLoaded` hands the bridge over explicitly and runs inside the SwiftUI update: attach at once,
/// defer `play()` and any `@State` write to the next main-actor turn.
@available(iOS 17.0, macCatalyst 17.0, *)
@MainActor
@Observable
public final class TrideKitScene {

    /// Rust runtime handle, owned for the life-time of this object.
    /// uniffi releases on `deinit`. `nonisolated(unsafe)`: the core serialises every call with its
    /// own lock, so the runtime-only calls reach it from any thread (`ThreadRule.swift`).
    nonisolated(unsafe) let trideScene: TrideScene

    /// The sound the scene hears, by input (`TrideKitScene+Audio`).
    @ObservationIgnored let audio = SceneAudioInputs()

    /// The app's pictures by `(name, revision)` (`TrideKitScene+Images`) and the latest request
    /// per image input.
    @ObservationIgnored let hostImages = HostImageStore()
    @ObservationIgnored var imageRequests = HostImageRequests()

    let dispatcher: PointerDispatcher

    /// Scene signals of the running tick, drained after dispatch.
    private let signals = SceneSignalInbox()
    @ObservationIgnored
    private var eventHandlers: [(id: UInt64, handler: @MainActor (TrideKitEvent) -> Void)] = []
    @ObservationIgnored
    private var stateHandlers: [(id: UInt64, handler: @MainActor (TrideKitStateChange) -> Void)] = []
    @ObservationIgnored
    private var diagnosticHandlers: [(id: UInt64, handler: @MainActor (TrideKitDiagnostic) -> Void)] = []
    /// Problems waiting for the handlers (delivered with the tick's signals).
    @ObservationIgnored
    private var pendingDiagnostics: [TrideKitDiagnostic] = []
    /// Problems already reported — each `(kind, subject, message)` once per scene.
    @ObservationIgnored
    private var reportedDiagnostics: Set<String> = []
    @ObservationIgnored
    private var nextSubscriptionId: UInt64 = 1

    @ObservationIgnored
    private weak var bridge: FilamentBridge?
    /// The view's size in points for the pointer; `.zero` until the view reports it. Observable.
    public private(set) var viewportSize: CGSize = .zero
    /// What the last `set_viewport` carried: a tick sends it again only when the view or its camera moved.
    @ObservationIgnored
    private var sentViewport: (size: CGSize, camera: TrideHostCamera?)?
    /// A fast-forward's final state that ran before the bridge was attached — dispatched first on
    /// `attachBridge`: mutations are deltas, a later tick would not repeat them.
    @ObservationIgnored
    private var pendingAdvance: [TrideMutation] = []

    @ObservationIgnored
    private var tickDriver: TickDriver?
    /// The showing view's render rate: the tick loop wakes at most this often; `nil` = every
    /// display frame. Set by the view, kept across `pause()` / `play()`.
    @ObservationIgnored
    private(set) var wakeRate: Int?

    /// `true` while the scene's clock runs (`play()` … `pause()`). Observable: a SwiftUI view that reads
    /// it updates when it changes.
    public private(set) var isPlaying: Bool = false

    /// Why the scene is drawn only in part — a `requires` iOS does not draw yet, a material without
    /// values the record does not carry yet; `nil` = drawn whole. The same text went to the
    /// diagnostics (`partialDraw`).
    @_spi(TrideTesting) @ObservationIgnored
    public private(set) var partialDrawNotice: String?

    /// The scene as the view built it from the core's record — the dispatcher addresses nodes,
    /// lights and cameras through it. `nil` until a view built it.
    @ObservationIgnored
    private(set) var recordScene: RecordScene?

    /// Load a `.trd` on the calling (main) actor. For a large scene prefer `load(_:)`, which reads the
    /// archive off the main thread.
    ///
    /// - Throws: `TrideError` for an archive the runtime cannot read or a scene it refuses (e.g. one
    ///   from a newer editor).
    public convenience init(trdBytes: Data) throws {
        try self.init(parsed: Self.parse(trdBytes))
    }

    /// Load a `.trd`, reading the archive and building the runtime off the main thread.
    public static func load(_ trdBytes: Data) async throws -> TrideKitScene {
        let parsed = try await Task.detached(priority: .userInitiated) { try parse(trdBytes) }.value
        return try TrideKitScene(parsed: parsed)
    }

    /// What loading produces before the scene is wired on the main actor. The runtime handle is safe to
    /// hand over: the core guards it with its own lock.
    private struct Parsed: @unchecked Sendable {
        let scene: TrideScene
        let bytes: Int
    }

    /// The runtime reads the archive once — TrideKit has no zip reader of its own.
    nonisolated private static func parse(_ trdBytes: Data) throws -> Parsed {
        Parsed(scene: try TrideScene.load(trdBytes: trdBytes), bytes: trdBytes.count)
    }

    private init(parsed: Parsed) throws {
        self.trideScene = parsed.scene
        // Only the pointers whose value moved; the bridge holds the rest.
        try trideScene.setChangedOnly(enabled: true)
        self.dispatcher = PointerDispatcher(scene: trideScene)
        dispatcher.report = { [weak self] in self?.note($0) }
        dispatcher.hostImages = hostImages
        _ = try trideScene.subscribeEvents(listener: signals)
        _ = try trideScene.subscribeLayerStateChanges(listener: signals)
        logger.notice(
            "loaded TrideScene from \(parsed.bytes) bytes"
        )
    }

    /// Manifest's view-model bindings as seen by the Rust runtime —
    /// includes regular bindings (`number`/`bool`/`vec3`/`color`/
    /// `enum`) and triggers (`kind == "trigger"`). Used by hosts that
    /// want to validate a custom-authored `.trd` declares the keys
    /// their UI calls into. Any thread: it only asks the runtime, like `validateBindings`.
    nonisolated public func availableBindings() -> [TrideKitBinding] {
        trideScene.listBindings().map(TrideKitBinding.init)
    }

    /// The binding's current value. Throws `TrideError.InvalidBinding` for a name the scene does not
    /// declare.
    nonisolated public func getBinding(_ name: String) throws -> TrideValue {
        try trideScene.getBinding(name: name)
    }

    /// The render settings as the bridge takes them (`FilamentBridge.applyRenderSettings`), the host's
    /// knobs applied: a test host copies it, changes one setting and applies the variant (TrideTestApp's
    /// edge-flicker probe). `nil` when the scene authors none and no knob is set.
    @_spi(TrideTesting) public func renderSettingsDictionary() -> [String: Any]? {
        TrideKitThreadRule.requireMain()
        return appliedRenderSettings().map(RenderSettingsDictionary.make)
    }

    /// The host's render knobs over the scene's render settings; `nil` = the scene's settings exactly.
    /// Observable. Set with `setRenderOverrides(_:)` or a view's `renderOverrides`.
    public private(set) var renderOverrides: TrideRenderOverrides?

    /// The render settings the scene's designer chose (every default resolved), whatever the knobs
    /// say: what `renderOverrides` departs from. `nil` when the scene authors none. Any thread.
    nonisolated public var sceneRenderSettings: TrideRenderSettings? {
        trideScene.renderSettings()
    }

    /// Lay the host's render knobs over the scene's render settings — e.g. MSAA and bloom off while
    /// the phone runs hot. Every `nil` field keeps the scene's value; `nil` (or no field set) returns
    /// to the scene's settings exactly. Bloom, shadows and depth of field only switch off (`true`
    /// keeps the scene's); MSAA goes both ways, with 2, 4 or 8 samples (another count keeps the
    /// scene's, with one warning). Takes effect from the next frame on the same view, the scene's
    /// state and animations untouched; the knobs are not saved to the file. Before the scene is
    /// shown they wait for its view.
    public func setRenderOverrides(_ overrides: TrideRenderOverrides?) {
        TrideKitThreadRule.requireMain()
        let knobs = overrides == TrideRenderOverrides() ? nil : overrides
        guard knobs != renderOverrides else { return }
        renderOverrides = knobs
        guard let bridge, !bridge.isDisposed else { return }
        applyRenderSettings(to: bridge)
        // Re-applying returned the animated settings to the scene's own: the core's whole frame
        // hands out what its animation has them at.
        handOutWholeFrame()
        bridge.setNeedsRender()
    }

    /// Per generator with module symbols (cloner clones, L-system modules): how many instances the
    /// core evaluated and how many the attached bridge draws — test / debug introspection (a device
    /// check reads «all 40 000 drawn» instead of guessing by eye). `drawn` is 0 without a bridge.
    @_spi(TrideTesting) public func moduleInstanceCounts() -> [TrideModuleInstanceCount] {
        TrideKitThreadRule.requireMain()
        return trideScene.listGenerators().compactMap { generator in
            guard !generator.modules.isEmpty else { return nil }
            let evaluated = generator.modules.reduce(0) { sum, module in
                sum + trideScene.generatorModuleTransforms(id: generator.id, symbol: module.symbol).count / 64
            }
            let drawn = bridge?.moduleInstanceCount(generator.id) ?? 0
            return TrideModuleInstanceCount(generator: generator.id, evaluated: evaluated, drawn: drawn)
        }
    }

    /// Verify that the loaded manifest declares every binding/trigger
    /// in `required`. Throws `MissingBindingsError` listing the missing
    /// keys so hosts can surface a friendly error UI rather than a
    /// failure on the first `setBinding` (it throws `InvalidBinding` on
    /// an undeclared name).
    ///
    /// The check is name-only; binding `kind` matching is **not**
    /// asserted here (a host swapping `progress` from `number` to
    /// `vec3` would still pass — that's a per-app concern). Add a
    /// per-app type check on top if needed.
    nonisolated public func validateBindings(_ required: [String]) throws {
        let declared = Set(trideScene.listBindings().map { $0.name })
        let missing = required.filter { !declared.contains($0) }
        guard missing.isEmpty else {
            throw MissingBindingsError(missing: missing, declared: Array(declared).sorted())
        }
    }

    /// What the scene draws only in part: added to `partialDrawNotice` (the build's reasons joined
    /// by `; `, as Android) and said once to the diagnostics.
    func notePartialDraw(_ notice: String) {
        partialDrawNotice = partialDrawNotice.map { "\($0); \(notice)" } ?? notice
        note(TrideKitDiagnostic(kind: .runtime, subject: "partialDraw", message: notice))
    }

    /// Build the scene in `bridge` from the core's scene record — what `FilamentMetalView` does. A
    /// scene the core hands no record of is not built: `TrideKitSceneNotBuiltError`.
    @_spi(TrideTesting) public func load(into bridge: FilamentBridge) throws {
        guard let record = trideScene.sceneRecord() else {
            let error = TrideKitSceneNotBuiltError(reason: "the runtime handed no scene record")
            note(TrideKitDiagnostic(kind: .load, subject: "scene", message: error.message))
            throw error
        }
        try loadByRecord(record, into: bridge)
    }

    /// Build the scene in `bridge` from the core's `record` (`RecordScene`). The dispatcher binds to
    /// its table at `attachBridge`, and the core says each node's visibility to the renderer
    /// (`/node/{uid}/drawVisible`). A `requires` iOS does not draw is said once (`partialDraw`).
    @_spi(TrideTesting) public func loadByRecord(_ record: TrideSceneRecord, into bridge: FilamentBridge) throws {
        let built = RecordScene(record: record)
        // A new build says its own reasons only.
        partialDrawNotice = nil
        try bridge.loadScene(places: built.places)
        recordScene = built
        if let notice = RecordScene.requiresNotice(record.requires) { notePartialDraw(notice) }
        do {
            try trideScene.setDrawVisibility(enabled: true)
        } catch {
            note(TrideKitDiagnostic(kind: .runtime, subject: "drawVisibility", message: error.localizedDescription))
        }
    }

    /// Once the meshes are built: the materials drawn without values the record does not carry yet
    /// (`FilamentBridge.standardMaterialGaps`), by slug, said once — such a frame is drawn in part.
    /// The same text as Android's `RecordScene.gapsNotice`.
    static func recordGapsNotice(_ gaps: [String: [String]]) -> String? {
        guard !gaps.isEmpty else { return nil }
        let each = gaps.keys.sorted().map { "'\($0)' without its \(gaps[$0, default: []].joined(separator: ", "))" }
        return "the scene draws " + each.joined(separator: "; ")
    }

    /// Each node's, light's and camera's place and its world matrix composed from the record in
    /// `Double` (`RecordScene.worlds`), by node uid and light / camera id; `[:]` before a view. For
    /// checks against what Filament composed (`FilamentBridge.worldTransform(atIndex:)`).
    @_spi(TrideTesting) public func recordWorlds() -> [String: (place: Int, world: [Double])] {
        guard let recordScene else { return [:] }
        let worlds = RecordScene.worlds(of: recordScene.places)
        var out: [String: (place: Int, world: [Double])] = [:]
        for (id, place) in recordScene.placeOf.merging(recordScene.carrierPlace, uniquingKeysWith: { node, _ in node }) {
            out[id] = (place, worlds[place])
        }
        return out
    }

    /// The palette check (the parity gate): every palette the core handed last against the one
    /// gltfio's `updateBoneMatrices` rule makes of the worlds the attached view's TransformManager
    /// composed (`BonePaletteCheck`); `nil` before `attachBridge`.
    @_spi(TrideTesting) public func bonePaletteCheck() -> BonePaletteCheck.Result? {
        guard let bridge, !bridge.isDisposed, let recordScene else { return nil }
        // A skin cut into parts: each part's palette names its own joints (the decoder's cut), as
        // the meshes were drawn — read at upload, so a released source changes nothing here.
        return BonePaletteCheck.measure(
            palettes: Array(dispatcher.lastBonePalettes.values), skins: recordScene.skins,
            partJoints: dispatcher.skinPartJoints, world: { bridge.worldTransform(atIndex: $0)?.map(\.floatValue) })
    }

    /// The bone palette the core handed last per mesh row and skin part, ordered by both — what the
    /// attached view bends its skins with; `[]` before `attachBridge`.
    @_spi(TrideTesting) public func bonePalettes() -> [TrideBonePalette] {
        dispatcher.lastBonePalettes.keys.sorted().compactMap { dispatcher.lastBonePalettes[$0] }
    }

    /// The world matrix of every node, light and camera the attached view draws, as its
    /// TransformManager composes it (16 numbers, column-major) — by node uid and light / camera id;
    /// one the view has no entity for is left out. For checks (the parity gate); `[:]` before
    /// `attachBridge`.
    @_spi(TrideTesting) public func nodeWorlds() -> [String: [Float]] {
        guard let bridge, !bridge.isDisposed else { return [:] }
        return dispatcher.worlds(on: bridge)
    }

    /// Wire the scene to the bridge its view built it in (`load(into:)`; `FilamentMetalView`'s
    /// `onLoaded`). Another bridge later takes the scene over (a view shown again). A bridge the
    /// scene was not built in is addressed by the core's record all the same.
    @_spi(TrideTesting) public func attachBridge(_ bridge: FilamentBridge) {
        TrideKitThreadRule.requireMain()
        // One view shows one scene: a view still showing it stops following it from here on.
        if let shown = self.bridge, shown !== bridge, !shown.isDisposed {
            note(TrideKitDiagnostic(
                kind: .runtime, subject: "view",
                message: "the scene is already shown in another view — one view shows one scene, so that view stops following it; give each view its own TrideKitScene"))
        }
        self.bridge = bridge
        // The pixelation pass lays its blocks where the core's grid says.
        bridge.pixelationGridSource = Self.corePixelationGrid
        // The default camera stands where the core frames the scene for the view's aspect.
        bridge.defaultFramingSource = coreDefaultFraming()
        // The table the scene was built by (`load(into:)`); a bridge a host built itself is
        // addressed by the same table, read from the core's record.
        if let table = recordScene ?? trideScene.sceneRecord().map(RecordScene.init(record:)) {
            dispatcher.bind(to: bridge, record: table)
        } else {
            note(TrideKitDiagnostic(
                kind: .load, subject: "scene", message: "scene not built: the runtime handed no scene record"))
        }
        if recordScene != nil, let notice = Self.recordGapsNotice(bridge.standardMaterialGaps) {
            notePartialDraw(notice)
        }
        releaseUnshownPictures()
        bridgeTookRenderSettings = false
        applyRenderSettings(to: bridge)
        applyEnvironment(to: bridge)
        if !pendingAdvance.isEmpty {
            dispatcher.dispatch(mutations: pendingAdvance, bridge: bridge)
            pendingAdvance = []
        }
        // Whatever an earlier tick wrote without a renderer arrives again.
        handOutWholeFrame()
        // The first frame, whatever the first tick moved (render settings / environment too).
        bridge.setNeedsRender()
    }

    /// Orbit the bridge's default viewport camera around the framed asset
    /// (`yaw` / `pitch` radians; `distanceScale` × the framing distance, 1 = as
    /// framed). No-op before `attachBridge` or while an authored camera is active
    /// — that camera is driven through its `/cameras/{id}` TRS instead.
    @_spi(TrideTesting) public func setViewCameraOrbit(yaw: Float, pitch: Float, distanceScale: Float) {
        TrideKitThreadRule.requireMain()
        bridge?.setViewCameraOrbit(yaw: yaw, pitch: pitch, distanceScale: distanceScale)
        bridge?.setNeedsRender()
    }

    // MARK: - Pointer input

    /// The view's size in points. While no manifest camera draws, the bridge's
    /// default camera goes with it, so the scene picks through the camera the user sees.
    public func setViewport(size: CGSize) {
        TrideKitThreadRule.requireMain()
        viewportSize = size
        pushViewport()
    }

    /// Queue a pointer event at `point` (view points from the top-left). The scene picks what is
    /// under it and runs its listeners — taps, drags — on the next tick.
    public func pointer(_ phase: TridePointerPhase, at point: CGPoint) {
        TrideKitThreadRule.requireMain()
        pushViewport()
        do {
            try trideScene.pointer(phase: phase, x: Float(point.x), y: Float(point.y))
        } catch {
            note(TrideKitDiagnostic(kind: .runtime, subject: "pointer", message: error.localizedDescription))
            deliverDiagnostics()
        }
    }

    /// Bind source `source` to the scene's number input `input`: from the next tick the runtime
    /// writes it from the source, mapped `mapping` (`nil`: the speed 0…2 screens / s, the pass
    /// 0…1, onto 0…1), and `setBinding` on it is refused until `unbindInput`. Binding it again
    /// replaces the binding; `reset()` keeps it, a new scene starts with none. A refusal (an input
    /// the scene lacks, not a number, computed or driven by the scene's own pointer interactions,
    /// an empty or non-finite range) throws.
    nonisolated public func bindInput(
        _ input: String,
        to source: TrideInputSource,
        mapping: TrideInputMapping? = nil
    ) throws {
        try trideScene.bindInput(input: input, source: source, mapping: mapping)
    }

    /// The app writes input `input` again; it keeps its value. `true` when it was bound.
    @discardableResult
    nonisolated public func unbindInput(_ input: String) -> Bool {
        (try? trideScene.unbindInput(input: input)) ?? false
    }

    /// The app's bindings, in the order they were made.
    nonisolated public func boundInputs() -> [TrideBoundInput] {
        trideScene.boundInputs()
    }

    /// Your scroll view, in points: how far it is scrolled (`offset`), its visible height
    /// (`extent`) and where the scene sits in it (`sceneTop`, `sceneHeight`). Call it whenever any of
    /// them changes while a scroll source is bound (`bindInput`); the bound inputs read the speed
    /// and the scene's pass through the view on the next tick. A value that is not finite, an
    /// `extent` of 0 or less or a negative `sceneHeight` is refused with a diagnostic, and the last
    /// report stays.
    public func scroll(offset: Double, extent: Double, sceneTop: Double, sceneHeight: Double) {
        TrideKitThreadRule.requireMain()
        do {
            try trideScene.scroll(offset: offset, extent: extent, sceneTop: sceneTop, sceneHeight: sceneHeight)
        } catch {
            note(TrideKitDiagnostic(kind: .runtime, subject: "scroll", message: error.localizedDescription))
            deliverDiagnostics()
        }
    }

    /// The directions the scene's drag interactions claim — `y`: it wants vertical swipes too,
    /// which a scroll view around it would otherwise take.
    nonisolated public func dragAxes() -> TrideDragAxes {
        trideScene.dragAxes()
    }

    /// Every node under `point`, nearest first — empty when no listener of the scene targets a node.
    public func pick(at point: CGPoint) -> [TrideKitHit] {
        TrideKitThreadRule.requireMain()
        pushViewport()
        return trideScene.pick(x: Float(point.x), y: Float(point.y)).map(TrideKitHit.init)
    }

    /// The view size and, while the bridge draws through its default camera, that camera.
    /// `onlyIfMoved`: skipped when both equal what the last call sent (the tick's check).
    private func pushViewport(onlyIfMoved: Bool = false) {
        guard viewportSize.width > 0, viewportSize.height > 0 else { return }
        var camera: TrideHostCamera?
        if let bridge, bridge.activeCameraId == nil {
            var position = simd_float3()
            var rotation = simd_float4(0, 0, 0, 1)
            var yfov: Float = 0
            bridge.defaultCamera(position: &position, rotation: &rotation, yfov: &yfov)
            camera = TrideHostCamera(
                x: position.x, y: position.y, z: position.z,
                qx: rotation.x, qy: rotation.y, qz: rotation.z, qw: rotation.w,
                yfov: yfov
            )
        }
        if onlyIfMoved, let sent = sentViewport, sent.size == viewportSize, sent.camera == camera { return }
        do {
            try trideScene.setViewport(
                width: Float(viewportSize.width), height: Float(viewportSize.height), camera: camera
            )
            sentViewport = (viewportSize, camera)
        } catch {
            note(TrideKitDiagnostic(kind: .runtime, subject: "setViewport", message: error.localizedDescription))
            deliverDiagnostics()
        }
    }

    /// Set a view-model binding value. Mirror of `TrideScene.setBinding(name:value:)`
    /// — kept here only to expose the API on the high-level wrapper so
    /// callers don't have to reach into `trideScene` directly.
    ///
    /// Setting a value does **not** immediately push mutations; the
    /// next `tick()` (from the CADisplayLink driver) will see the new
    /// value and emit dirty pointers as appropriate.
    nonisolated public func setBinding(_ name: String, _ value: TrideValue) throws {
        try trideScene.setBinding(name: name, value: value)
    }

    /// Every input of the scene with its limits and labels, and the values `setInputs` takes, as JSON:
    /// `{"inputs":[{name, kind, value, default, options?, min?, max?, maxLength?, multiline?,
    /// computed?, doc?, settings?, missingGlyphs?}], "values":{name: value}}` — the bytes every
    /// host gets. An input with `settings` is the one a settings screen shows; `kind` `"string"` is text.
    nonisolated public func inputsJSON() throws -> String {
        try trideScene.inputsJson()
    }

    /// Write a batch of input values at once. `json` is an object of input names and values
    /// (`{"greeting":"Hi","amount":7500}`); the scene checks them all and the good ones land together
    /// for the next `tick()`. A value outside its limits is refused with a code, never cut or
    /// clamped (the input keeps its text), and listed in the report. With `strict` nothing is written
    /// unless every value is good. A picture for an image input goes through `setImage`, not here.
    /// Throws `TrideError.InvalidBinding` when `json` is not a JSON object (nothing is written).
    @discardableResult
    nonisolated public func setInputs(json: String, strict: Bool = false) throws -> TrideKitInputsReport {
        let report = try trideScene.setInputsJson(valuesJson: json, strict: strict)
        return try JSONDecoder().decode(TrideKitInputsReport.self, from: Data(report.utf8))
    }

    /// Fire a manifest trigger (`start`, `reset`, …).
    nonisolated public func trigger(_ name: String) throws {
        try trideScene.trigger(name: name)
    }

    /// Read the value the core holds at `pointer` (JSON-encoded): a node's pose, visibility or
    /// weights from its node table (`/node/{uid}/…`), or scene state. `nil` when the pointer
    /// names nothing the core holds.
    @_spi(TrideTesting) nonisolated public func readProperty(_ pointer: String) -> String? {
        trideScene.readProperty(pointer: pointer)
    }

    /// Current state of the LAST state-machine layer; per-layer state is
    /// `layerCurrentState(at:)` / `currentState(ofLayer:)`.
    nonisolated public func currentState() -> String {
        trideScene.currentState()
    }

    /// Number of state-machine layers (`0` for an SM-less scene).
    nonisolated public var layerCount: Int {
        Int(trideScene.layerCount())
    }

    /// Slug of the layer at `index` — what `TrideKitStateChange.layerId` carries.
    nonisolated public func layerId(at index: Int) -> String? {
        guard let i = UInt32(exactly: index) else { return nil }
        return trideScene.layerId(index: i)
    }

    /// Human-readable name of the layer at `index`.
    nonisolated public func layerName(at index: Int) -> String? {
        guard let i = UInt32(exactly: index) else { return nil }
        return trideScene.layerName(index: i)
    }

    /// Current state of the layer at `index`.
    nonisolated public func layerCurrentState(at index: Int) -> String? {
        guard let i = UInt32(exactly: index) else { return nil }
        return trideScene.layerCurrentState(index: i)
    }

    /// Current state of the layer whose slug is `layerId`.
    nonisolated public func currentState(ofLayer layerId: String) -> String? {
        (0..<layerCount).first { self.layerId(at: $0) == layerId }.flatMap { layerCurrentState(at: $0) }
    }

    /// Receive the events the scene reports — on the main actor, right after the
    /// tick's mutations went to the bridge, so the host reacts in the same frame. Handlers run in
    /// registration order; a handler must not tick the scene.
    @discardableResult
    public func subscribeEvents(
        _ handler: @escaping @MainActor (TrideKitEvent) -> Void
    ) -> TrideKitSubscription {
        TrideKitThreadRule.requireMain()
        let token = makeSubscription()
        eventHandlers.append((token.id, handler))
        return token
    }

    /// Receive every layer's transitions, with the layer — same delivery as
    /// `subscribeEvents`, interleaved with the events in the order the runtime produced them.
    @discardableResult
    public func subscribeStateChanges(
        _ handler: @escaping @MainActor (TrideKitStateChange) -> Void
    ) -> TrideKitSubscription {
        TrideKitThreadRule.requireMain()
        let token = makeSubscription()
        stateHandlers.append((token.id, handler))
        return token
    }

    /// Receive the problems the scene runs into (`TrideKitDiagnostic`) — on the main actor, after the
    /// tick that met them, or right after the call that failed. The log gets them too.
    @discardableResult
    public func subscribeDiagnostics(
        _ handler: @escaping @MainActor (TrideKitDiagnostic) -> Void
    ) -> TrideKitSubscription {
        TrideKitThreadRule.requireMain()
        let token = makeSubscription()
        diagnosticHandlers.append((token.id, handler))
        return token
    }

    /// Stop a `subscribeEvents` / `subscribeStateChanges` / `subscribeDiagnostics` handler. Unknown
    /// tokens are ignored.
    public func unsubscribe(_ subscription: TrideKitSubscription) {
        TrideKitThreadRule.requireMain()
        eventHandlers.removeAll { $0.id == subscription.id }
        stateHandlers.removeAll { $0.id == subscription.id }
        diagnosticHandlers.removeAll { $0.id == subscription.id }
    }

    /// Start the CADisplayLink tick loop. Idempotent — re-calling while
    /// already playing is a no-op. The first frame's `dt` is clamped at
    /// 16.67 ms to avoid a freshly-started loop emitting a giant first
    /// step, and the runtime is told its loop resumes (the scroll speed is
    /// measured afresh).
    public func play() {
        TrideKitThreadRule.requireMain()
        guard !isPlaying else { return }
        // The loop stood (or never ran): the scroll view's move since then is no speed.
        do {
            try trideScene.resume()
        } catch {
            note(TrideKitDiagnostic(kind: .runtime, subject: "resume", message: error.localizedDescription))
            deliverDiagnostics()
        }
        let driver = TickDriver { [weak self] dtMs in
            self?.tickAndDispatch(dtMs: dtMs)
        }
        driver.setPreferredRate(wakeRate)
        driver.start()
        tickDriver = driver
        isPlaying = true
        logger.debug("play()")
    }

    /// The view's render rate reaches the tick loop, at once when playing: the core is woken
    /// at most `fps` times a second and steps its fixed 60 Hz over the time since the last wake.
    func setWakeRate(_ fps: Int?) {
        guard wakeRate != fps else { return }
        wakeRate = fps
        tickDriver?.setPreferredRate(fps)
    }

    /// The rate the running tick loop wakes at; `nil` when paused or waking every display frame.
    var tickLoopRate: Int? { tickDriver?.preferredRate }

    /// Steps the scene ran (instrumentation: the rate tests count the loop's wakes).
    @ObservationIgnored
    private(set) var tickCount = 0

    /// Stop the CADisplayLink tick loop. Runtime keeps its current
    /// state — `play()` resumes from where we left off. For a *clean*
    /// reset, call `reset()`.
    public func pause() {
        TrideKitThreadRule.requireMain()
        tickDriver?.stop()
        tickDriver = nil
        isPlaying = false
        // No next tick is coming: a trunk the last one deferred is shown now.
        if let bridge, dispatcher.flushDeferredTrunks(into: bridge) { bridge.setNeedsRender() }
        logger.debug("pause()")
    }

    /// Return the scene to how it loaded — the initial states, every binding at its declared default,
    /// the glTF's own pose — and run one zero-time step, so the initial `onEnter` actions run and their
    /// events reach the handlers like any tick's. A trigger the scene names `reset` is an ordinary
    /// trigger: fire it with `trigger("reset")`.
    public func reset() throws {
        TrideKitThreadRule.requireMain()
        try trideScene.reset()
        // The core hands out its whole frame after a reset: what it no longer animates goes back to
        // the scene's own value.
        applyCurrentState(complete: true)
    }

    /// Single-shot tick + dispatch. Public so smoke tests and unit
    /// tests can drive the scene without a runloop.
    public func tickOnce(dtMs: Float) {
        TrideKitThreadRule.requireMain()
        tickAndDispatch(dtMs: dtMs)
    }

    /// Replay a host script and show where it ends: the timer a
    /// host would drive — `{deltaMs, inputs}` ticks, `{until, maxMs}` waits for a scene event —
    /// run by the Rust player in one call, its final state dispatched like one tick. The scene's
    /// clock continues from there. A script the runtime cannot read stops at step 0 with the reason;
    /// it throws only when the runtime itself fails.
    @discardableResult
    public func advance(scriptJSON: String) throws -> TrideKitAdvance {
        TrideKitThreadRule.requireMain()
        audio.hear(trideScene) { note($0) }
        pushViewport(onlyIfMoved: true)
        let result = try trideScene.advance(scriptJson: scriptJSON)
        for diagnostic in result.tick.diagnostics { note(evaluationDiagnostic(diagnostic)) }
        if let bridge {
            if !result.tick.mutations.isEmpty {
                dispatcher.dispatch(mutations: result.tick.mutations, bridge: bridge)
                dispatcher.flushDeferredTrunks(into: bridge)
                bridge.setNeedsRender()
            }
            if !result.tick.bonePalettes.isEmpty {
                dispatcher.applyBonePalettes(result.tick.bonePalettes, to: bridge)
                bridge.setNeedsRender()
            }
        } else {
            // Without a renderer the palettes go: `attachBridge` asks the core for its whole frame.
            pendingAdvance += result.tick.mutations
        }
        releaseUnshownPictures()
        deliverSignals()
        logger.notice(
            "advance: ticks=\(result.ticks, privacy: .public) mutations=\(result.tick.mutations.count, privacy: .public)"
        )
        guard let step = result.stoppedStep else { return .completed(ticks: Int(result.ticks)) }
        return .stopped(step: Int(step), reason: result.stoppedReason ?? "stopped", ticks: Int(result.ticks))
    }

    // MARK: - Private

    /// Materialize the authored HDR environment onto the
    /// freshly-attached bridge. Reads the schema-resolved descriptor through the
    /// Rust core (`TrideScene.environment()` — never parses `manifest.json`
    /// Swift-side), resolves the IBL/skybox KTX1
    /// bytes, and forwards them to `FilamentBridge.setIndirectLight`.
    ///
    /// Lives here rather than in `PointerDispatcher` (where lights materialize)
    /// because the custom-env KTX bytes come from *this* object's `.trd`
    /// resources (`assets/ibl/<uid>_*.ktx`), whereas presets come from the
    /// SDK-bundled `runtime-assets`. This is a static at-load apply; live
    /// rotation/intensity keyframes flow separately through tick mutations on
    /// `/environment_settings/*`. A scene that authored no
    /// environment keeps the background skybox — the manifest's colour, which
    /// `PointerDispatcher.bind(to:)` applied.
    private func applyEnvironment(to bridge: FilamentBridge) {
        guard let env = trideScene.environment() else { return }

        let iblData: Data?
        let skyboxData: Data?
        switch env.kind {
        case "preset":
            guard let preset = TrideEnvironmentPreset(rawValue: env.name) else {
                note(TrideKitDiagnostic(kind: .load, subject: env.name, message: "unknown environment preset — the default light stays"))
                return
            }
            iblData = TrideEnvironmentPresetLoader.iblData(preset)
            skyboxData = env.backgroundVisible ? TrideEnvironmentPresetLoader.skyboxData(preset) : nil
        case "custom":
            iblData = resourceData(named: "assets/ibl/\(env.name)_ibl.ktx")
            skyboxData = env.backgroundVisible ? resourceData(named: "assets/ibl/\(env.name)_skybox.ktx") : nil
        default:
            note(TrideKitDiagnostic(kind: .load, subject: env.name, message: "unknown environment kind '\(env.kind)' — the default light stays"))
            return
        }

        guard let ibl = iblData else {
            note(TrideKitDiagnostic(kind: .load, subject: env.name, message: "the environment's light map is missing — the default light stays"))
            return
        }

        // Euler XYZ radians → matrix happens at the bridge boundary;
        // the core stores Euler and never converts. The descriptor always
        // carries a length-3 vector, but guard defensively against a malformed
        // FFI payload rather than trapping on an index.
        let r = env.rotation
        let euler = r.count == 3
            ? SIMD3<Float>(Float(r[0]), Float(r[1]), Float(r[2]))
            : SIMD3<Float>(repeating: 0)
        bridge.setIndirectLight(
            ibl,
            skybox: skyboxData,
            intensity: Float(env.intensity),
            rotationEulerXYZ: euler)
        logger.notice(
            "environment applied: \(env.kind, privacy: .public) '\(env.name, privacy: .public)' (skybox=\(skyboxData != nil), intensity=\(env.intensity))"
        )
    }

    /// The core's pixelation grid (`pixelation_grid`) in the bridge's shape — the one grid every
    /// host draws, whether the pass runs and its block targets.
    /// The core's default framing in the bridge's shape. A framing the core cannot give is noted
    /// once and handed back empty (`orbitDistance` 0): the bridge then keeps the camera's last pose.
    func coreDefaultFraming() -> FilamentBridgeFramingSource {
        let core = trideScene
        return { [weak self] aspect in
            do {
                let f = try core.defaultFraming(aspect: aspect)
                return FilamentBridgeFraming(
                    eye: SIMD3(f.eyeX, f.eyeY, f.eyeZ), target: SIMD3(f.targetX, f.targetY, f.targetZ),
                    yfov: f.yfov, znear: f.near, zfar: f.far, orbitDistance: f.orbitDistance)
            } catch {
                MainActor.assumeIsolated {
                    self?.note(TrideKitDiagnostic(
                        kind: .runtime, subject: "defaultFraming", message: error.localizedDescription))
                }
                return FilamentBridgeFraming()
            }
        }
    }

    static let corePixelationGrid: FilamentBridgePixelationGridSource = { width, height, size, pixelsPerPoint in
        let g = pixelationGrid(width: width, height: height, size: size, pixelsPerPoint: pixelsPerPoint)
        return FilamentBridgePixelationGrid(
            block: g.block, shiftX: g.shiftX, shiftY: g.shiftY, columns: g.columns, rows: g.rows,
            pixelates: g.pixelates, targetColumns: g.targetColumns, targetRows: g.targetRows)
    }

    /// A zero-time step after the core was asked for its whole frame: every value handed out before
    /// reaches the bridge again (a new bridge, or render settings re-applied over the animated ones).
    private func handOutWholeFrame() {
        do {
            try trideScene.requestFullFrame()
        } catch {
            note(TrideKitDiagnostic(kind: .runtime, subject: "requestFullFrame", message: error.localizedDescription))
        }
        applyCurrentState()
    }

    /// The attached bridge was handed render settings (it no longer draws what a new view draws).
    @ObservationIgnored
    private var bridgeTookRenderSettings = false

    /// The render settings to draw: the core lays the knobs over the scene's (no knob = the scene's
    /// exactly); `nil` for a scene without render settings and no knob.
    private func appliedRenderSettings() -> TrideRenderSettings? {
        trideScene.renderSettingsWith(overrides: renderOverrides ?? TrideRenderOverrides())
    }

    /// The scene's `renderSettings` with the host's knobs, as the core lays them (the editor viewport
    /// shows the same without knobs). The FFI hands back the normalized form (defaults resolved by
    /// the schema), projected 1:1 onto the bridge dictionary — no host-side defaults. Absent → the
    /// view draws as a new view does (Filament's defaults): one that took knobbed settings goes back
    /// to the options it was made with, the schema's defaults are not what a new view draws.
    private func applyRenderSettings(to bridge: FilamentBridge) {
        guard let settings = appliedRenderSettings() else {
            if bridgeTookRenderSettings {
                bridge.restoreInitialRenderSettings()
                bridgeTookRenderSettings = false
            }
            logger.notice("renderSettings: not authored — Filament defaults")
            return
        }
        bridge.applyRenderSettings(RenderSettingsDictionary.make(settings))
        bridgeTookRenderSettings = true
        logger.notice(
            "renderSettings applied (knobs \(self.renderOverrides == nil ? "none" : "set", privacy: .public)): aa=\(RenderSettingsDictionary.antiAliasing(settings.antiAliasing), privacy: .public) toneMapping=\(RenderSettingsDictionary.toneMapping(settings.toneMapping.operator), privacy: .public) exposure=\(settings.toneMapping.exposure) msaa=\(settings.msaa.enabled)/\(settings.msaa.sampleCount) shadows=\(settings.shadowingEnabled) dof=\(settings.depthOfField.enabled) bloom=\(settings.bloom.enabled) ssao=\(settings.ssao.enabled) ssr=\(settings.screenSpaceReflections.enabled)"
        )
    }

    /// The archive's resource `name` (e.g. `assets/ibl/<uid>_ibl.ktx`), or `nil` — through the runtime,
    /// the one reader of the archive.
    private func resourceData(named name: String) -> Data? {
        trideScene.assetBytes(uri: name)
    }

    private func applyCurrentState(complete: Bool = false) {
        tickAndDispatch(dtMs: 0, complete: complete)
        if bridge != nil {
            logger.debug("applyCurrentState: pushed initial state to Filament")
        }
    }

    /// Run a single runtime tick and (if a bridge is attached) push the
    /// resulting mutations to Filament.
    ///
    /// The runtime tick is **always** executed, even before the bridge
    /// has been attached — this lets headless smoke harnesses (e.g.
    /// `TrideKitSceneSmokeTest`) drive the scene without standing up a
    /// Metal context, and matches the lifecycle of host SwiftUI apps
    /// which may want to set inputs and tick a few frames during cold
    /// load before the renderer is on screen.
    ///
    /// `complete`: the core hands out its whole frame (after a reset) — see `PointerDispatcher`.
    private func tickAndDispatch(dtMs: Float, complete: Bool = false) {
        tickCount += 1
        // The trunks the previous tick re-evaluated get their meshes now (PointerDispatcher).
        if let bridge, dispatcher.flushDeferredTrunks(into: bridge) { bridge.setNeedsRender() }
        // What every input sounds like now — the step reads it.
        audio.hear(trideScene) { note($0) }
        // The camera the view draws through, before the step: tagged objects turn toward it this frame.
        pushViewport(onlyIfMoved: true)
        let result: TickResult
        do {
            let interval = signposts.beginInterval("tick")
            defer { signposts.endInterval("tick", interval) }
            result = try trideScene.tick(dtMs: dtMs)
        } catch {
            note(TrideKitDiagnostic(kind: .runtime, subject: "tick", message: error.localizedDescription))
            deliverDiagnostics()
            return
        }
        for diagnostic in result.diagnostics { note(evaluationDiagnostic(diagnostic)) }
        if let bridge, complete || !result.mutations.isEmpty {
            let interval = signposts.beginInterval("dispatch", "\(result.mutations.count) mutations")
            dispatcher.dispatch(mutations: result.mutations, bridge: bridge, complete: complete)
            // An on-demand view draws this frame; a tick that moved nothing draws none.
            bridge.setNeedsRender()
            signposts.endInterval("dispatch", interval)
            // A slot that moved to a newer picture or back to its placeholder frees the old pixels.
            releaseUnshownPictures()
        }
        // The skins' palettes this tick moved, after the poses they were made from.
        if let bridge, !result.bonePalettes.isEmpty {
            dispatcher.applyBonePalettes(result.bonePalettes, to: bridge)
            bridge.setNeedsRender()
        }
        // A paused scene (a single `tickOnce`) has no next tick to wait for.
        if !isPlaying, let bridge, dispatcher.flushDeferredTrunks(into: bridge) { bridge.setNeedsRender() }
        // After the tick's mutations reached the bridge — the host reacts this frame.
        deliverSignals()
    }

    /// Queue a problem for `subscribeDiagnostics` and log it — each `(kind, subject, message)` once per
    /// scene: a binding that fails every step is reported once, not sixty times a second.
    func note(_ diagnostic: TrideKitDiagnostic) {
        guard reportedDiagnostics.insert(diagnostic.description).inserted else { return }
        evalLogger.error("\(diagnostic.description, privacy: .public)")
        pendingDiagnostics.append(diagnostic)
    }

    func deliverDiagnostics() {
        let ready = pendingDiagnostics
        pendingDiagnostics.removeAll()
        for diagnostic in ready {
            for entry in diagnosticHandlers { entry.handler(diagnostic) }
        }
    }

    private func makeSubscription() -> TrideKitSubscription {
        defer { nextSubscriptionId += 1 }
        return TrideKitSubscription(id: nextSubscriptionId)
    }

    /// Hand this tick's scene signals to the handlers, in the runtime's order. Iterates over
    /// snapshots, so a handler may unsubscribe (itself or others) while being called.
    private func deliverSignals() {
        deliverDiagnostics()
        for signal in signals.drain() {
            switch signal {
            case .event(let event):
                for entry in eventHandlers { entry.handler(event) }
            case .stateChange(let change):
                for entry in stateHandlers { entry.handler(change) }
            }
        }
    }
}

@available(iOS 17.0, macCatalyst 17.0, *)
func evaluationDiagnostic(_ diagnostic: EvalDiagnostic) -> TrideKitDiagnostic {
    let suggestion = diagnostic.suggestion.map { " — \($0)" } ?? ""
    return TrideKitDiagnostic(
        kind: .evaluation,
        subject: diagnostic.pointer,
        message: "\(diagnosticKindName(diagnostic.kind)): \(diagnostic.message)\(suggestion)"
    )
}

@available(iOS 17.0, macCatalyst 17.0, *)
private func diagnosticKindName(_ kind: EvalDiagnosticKind) -> String {
    switch kind {
    case .unknownIdentifier:
        return "UnknownIdentifier"
    case .unknownFunction:
        return "UnknownFunction"
    case .typeMismatch:
        return "TypeMismatch"
    case .arityMismatch:
        return "ArityMismatch"
    case .arrayLengthMismatch:
        return "ArrayLengthMismatch"
    case .domainError:
        // `sqrt(-1)`, `mod(x, 0)` — a built-in outside its domain.
        return "DomainError"
    case .generator:
        // A procedural generator failed to evaluate (grammar / budget).
        return "Generator"
    }
}

/// Thrown by `TrideKitScene.validateBindings(_:)` when the loaded
/// `.trd` manifest does not declare one or more of the bindings /
/// triggers the host requires: a designer-authored scene must conform to
/// the host's binding contract or fail loudly.
public struct MissingBindingsError: LocalizedError, Equatable {
    /// Binding/trigger names the host requires but the manifest does
    /// not declare. Order preserved from the host's `required` list.
    public let missing: [String]
    /// All bindings the manifest *did* declare — surfaced so the
    /// error UI can hint at near-misses ("you wrote `progres`, did
    /// you mean `progress`?").
    public let declared: [String]

    public var errorDescription: String? {
        let missingList = missing.joined(separator: ", ")
        let declaredList = declared.isEmpty ? "(none)" : declared.joined(separator: ", ")
        return "Scene is missing required bindings: \(missingList). Declared: \(declaredList)."
    }
}

/// A view could not build the scene: the runtime handed it no scene record. `reason` is the
/// runtime's words; the scene itself stays usable (its state, bindings and signals).
public struct TrideKitSceneNotBuiltError: LocalizedError, Equatable {
    public let reason: String

    public var message: String { "scene not built: \(reason)" }

    public var errorDescription: String? { message }
}

#endif  // canImport(TrideKitFilamentBridge): a host without the bridge (macOS) gets TrideCore only
