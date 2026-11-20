/// The scene's frame prepared off the main thread: the tick, everything the dispatcher would
/// pull from the core for it, and the trunks' packed streams — made on one serial queue per process,
/// applied to the bridge on the main thread one display frame later.
///
/// ## Public API
/// - `TrideKitTickMode` — `.background` (the default: the tick on TrideKit's queue) or `.mainThread`
///   (the tick on the main thread in the same wake, as before 0.19.0 — kept for one release).
///
/// ## Internal API
/// - `ScenePreparation.queue` — the one serial queue (QoS `.userInteractive`) every scene of the
///   process prepares on, one frame at a time.
/// - `PrepareJob` — what the queue runs for one frame.
/// - `PreparedFrame` — one tick's self-contained frame: the result, the scene signals the tick
///   raised, the core reads (`CorePulls`); `wait()` blocks until it is done.
/// - `PullPlan` — what the dispatcher draws, copied on the main thread when the tick is queued:
///   which reads the frame will need.
/// - `CorePulls` — those reads, answered from the frame; a read the plan did not foresee goes to the
///   core (counted in `misses`).
///
/// ## Non-obvious constraints
/// - The queue touches the core and pure packing only (`FilamentBridge.packGeneratedGeometry`) —
///   never a Filament object, the view or the dispatcher's state.
/// - The main thread applies a frame before it queues the next tick (`TrideKitScene.wake`), so
///   while a frame is applied no tick runs: a read the plan missed sees the very tick being shown.
/// - Ticks are thread-agnostic in the core (R-FixedStep60, R-Determinism): the same `dt` and inputs
///   give the same mutations on either thread; `seal_inputs` on the main thread decides the tick an
///   input lands in.
///
/// ## Cross-refs
/// - `TrideKitScene` (wake / submit / apply / settle), `PointerDispatcher.pullPlan`.

import Foundation

#if canImport(TrideCore)
import TrideCore
#endif

/// Which thread ticks a scene.
public enum TrideKitTickMode: Sendable, Equatable {
    /// The tick and the core reads on TrideKit's preparation queue; the frame reaches the view one
    /// display frame after the wake that queued it. The default.
    case background
    /// The tick on the main thread, shown in the same wake — TrideKit before 0.19.0. Kept for one
    /// release as a way back (`TrideKitScene.defaultTickMode`); it goes in 0.20.0.
    case mainThread
}

#if canImport(TrideKitFilamentBridge)
import TrideKitFilamentBridge

/// The process's one preparation queue.
enum ScenePreparation {
    static let queue = DispatchQueue(label: "studio.tride.kit.scene-prepare", qos: .userInteractive)
}

/// What the dispatcher draws — the reads a frame will need, copied on the main thread.
struct PullPlan: @unchecked Sendable {
    /// Generator id → its module bindings (symbol, datablock).
    var modules: [String: [(symbol: String, datablock: String)]] = [:]
    /// `id\0symbol` of the modules one of whose batches samples the instance rows.
    var rowSymbols: Set<String> = []
    /// Datablocks registered as module meshes.
    var registered: Set<String> = []
    /// Trunk epoch on screen per generator.
    var uploadedTrunk: [String: UInt64] = [:]
    /// Generators drawn with pulses.
    var pulsed: Set<String> = []
    /// Nodes drawing a datablock: (node uid, datablock uid).
    var parametric: [(node: String, datablock: String)] = []
    /// Palette slug → its node-material families.
    var graphFamilies: [String: [String]] = [:]
    /// Slugs with photo slots.
    var slotSlugs: Set<String> = []
}

/// The core's reads for one frame, made after its tick on the preparation queue.
final class CorePulls: @unchecked Sendable {
    var geometry: [String: TrideGeneratedGeometry?] = [:]
    /// The trunks' streams packed for the bridge (`FilamentBridge.packGeneratedGeometry`).
    var packed: [String: NSDictionary] = [:]
    var transforms: [String: Data] = [:]
    var colors: [String: Data] = [:]
    var flows: [String: TridePulseTable?] = [:]
    var datablocks: [String: TrideDatablockMesh?] = [:]
    var deformed: [String: TrideDatablockMesh?] = [:]
    var environment: TrideEnvironmentFrame??
    var materialParams: [String: TrideMaterialParams?] = [:]
    var graphs: [String: TrideMaterialGraph?] = [:]
    var graphValues: [String: [TrideMaterialGraphValue]] = [:]
    /// Reads the frame did not carry, made on the core while applying (main thread).
    private(set) var misses = 0

    static func key(_ a: String, _ b: String) -> String { a + "\u{0}" + b }

    func generatedGeometry(_ id: String, core: TrideScene) -> TrideGeneratedGeometry? {
        if let hit = geometry[id] { return hit }
        misses += 1
        return core.generatedGeometry(id: id)
    }

    func moduleTransforms(_ id: String, _ symbol: String, core: TrideScene) -> Data {
        if let hit = transforms[Self.key(id, symbol)] { return hit }
        misses += 1
        return core.generatorModuleTransforms(id: id, symbol: symbol)
    }

    func moduleColors(_ id: String, _ symbol: String, core: TrideScene) -> Data {
        if let hit = colors[Self.key(id, symbol)] { return hit }
        misses += 1
        return core.generatorModuleColors(id: id, symbol: symbol)
    }

    func flow(_ id: String, core: TrideScene) -> TridePulseTable? {
        if let hit = flows[id] { return hit }
        misses += 1
        return core.generatorFlow(id: id)
    }

    func datablockMesh(_ uid: String, core: TrideScene) -> TrideDatablockMesh? {
        if let hit = datablocks[uid] { return hit }
        misses += 1
        return core.datablockMesh(uid: uid)
    }

    func deformedMesh(_ uid: String, core: TrideScene) -> TrideDatablockMesh? {
        if let hit = deformed[uid] { return hit }
        misses += 1
        return core.deformedMesh(nodeUid: uid)
    }

    func environmentFrame(core: TrideScene) -> TrideEnvironmentFrame? {
        if let hit = environment { return hit }
        misses += 1
        return core.environmentFrame()
    }

    func params(_ slug: String, core: TrideScene) -> TrideMaterialParams? {
        if let hit = materialParams[slug] { return hit }
        misses += 1
        return core.materialParams(slug: slug)
    }

    func graph(_ slug: String, _ family: String, core: TrideScene) -> TrideMaterialGraph? {
        if let hit = graphs[Self.key(slug, family)] { return hit }
        misses += 1
        return core.materialGraph(slug: slug, family: family)
    }

    func values(_ slug: String, _ family: String, core: TrideScene) -> [TrideMaterialGraphValue] {
        if let hit = graphValues[Self.key(slug, family)] { return hit }
        misses += 1
        return core.materialGraphValues(slug: slug, family: family)
    }

    /// Everything the dispatcher will read for `mutations` under `plan` — the pulls its
    /// `dispatch` makes, in the same order of need. Runs on the preparation queue.
    static func prefetch(_ mutations: [TrideMutation], plan: PullPlan, core: TrideScene) -> CorePulls {
        let pulls = CorePulls()
        for mutation in mutations {
            let pointer = mutation.pointer
            guard pointer.hasPrefix("/generators/") || pointer.hasPrefix("/datablocks/")
                || pointer.hasPrefix("/deformed/") || pointer.hasPrefix("/environments/")
                || pointer.hasPrefix("/materialOverrides/")
            else { continue }
            switch PointerDispatcher.classify(pointer) {
            case .generatorGeometry(let id):
                let trunk = PointerDispatcher.trunkEpoch(of: mutation.typedValue)
                if trunk == nil || plan.uploadedTrunk[id] != trunk, pulls.geometry[id] == nil {
                    let geometry = core.generatedGeometry(id: id)
                    pulls.geometry[id] = .some(geometry)
                    if let geometry, let packed = FilamentBridge.packGeneratedGeometry([
                        "positions": geometry.positions, "normals": geometry.normals,
                        "tangents": geometry.tangents, "colors": geometry.colors, "topology": geometry.topology,
                    ]) {
                        pulls.packed[id] = packed as NSDictionary
                    }
                }
                for module in plan.modules[id] ?? [] {
                    let key = Self.key(id, module.symbol)
                    if pulls.transforms[key] == nil {
                        pulls.transforms[key] = core.generatorModuleTransforms(id: id, symbol: module.symbol)
                    }
                    if plan.rowSymbols.contains(key), pulls.colors[key] == nil {
                        pulls.colors[key] = core.generatorModuleColors(id: id, symbol: module.symbol)
                    }
                    if !plan.registered.contains(module.datablock), pulls.datablocks[module.datablock] == nil {
                        pulls.datablocks[module.datablock] = .some(core.datablockMesh(uid: module.datablock))
                    }
                }
            case .generatorFlow(let id):
                if plan.pulsed.contains(id), pulls.flows[id] == nil {
                    pulls.flows[id] = .some(core.generatorFlow(id: id))
                }
            case .datablockGeometry(let uid):
                if pulls.datablocks[uid] == nil { pulls.datablocks[uid] = .some(core.datablockMesh(uid: uid)) }
                for node in plan.parametric where node.datablock == uid && pulls.deformed[node.node] == nil {
                    pulls.deformed[node.node] = .some(core.deformedMesh(nodeUid: node.node))
                }
            case .deformedGeometry(let uid):
                if pulls.deformed[uid] == nil { pulls.deformed[uid] = .some(core.deformedMesh(nodeUid: uid)) }
                for node in plan.parametric where node.node == uid && pulls.datablocks[node.datablock] == nil {
                    pulls.datablocks[node.datablock] = .some(core.datablockMesh(uid: node.datablock))
                }
            case .environment:
                if pulls.environment == nil { pulls.environment = .some(core.environmentFrame()) }
            case .materialOverride(let slug):
                if pulls.materialParams[slug] == nil { pulls.materialParams[slug] = .some(core.materialParams(slug: slug)) }
                for family in plan.graphFamilies[slug] ?? [] {
                    let key = Self.key(slug, family)
                    if pulls.graphValues[key] == nil {
                        pulls.graphValues[key] = core.materialGraphValues(slug: slug, family: family)
                    }
                    if plan.slotSlugs.contains(slug), pulls.graphs[key] == nil {
                        pulls.graphs[key] = .some(core.materialGraph(slug: slug, family: family))
                    }
                }
            default:
                continue
            }
        }
        return pulls
    }
}

/// What the queue needs for one frame. The core serialises every call with its own lock, the signal
/// inbox with its own; the plan is a copy.
struct PrepareJob: @unchecked Sendable {
    let frame: PreparedFrame
    let core: TrideScene
    let signals: SceneSignalInbox
    let plan: PullPlan?

    func run() { frame.prepare(core: core, signals: signals, plan: plan) }
}

/// One tick prepared on the queue — self-contained: applying it on the main thread asks the core
/// nothing it foresaw.
final class PreparedFrame: @unchecked Sendable {
    let dtMs: Float
    private let done = DispatchGroup()
    private let lock = NSLock()
    private var finished = false
    // Written on the queue before `finish`, read on the main thread after `wait` / `isDone`.
    private(set) var result: TickResult?
    private(set) var failure: Error?
    private(set) var signals: [SceneSignal] = []
    private(set) var pulls = CorePulls()
    /// Queue time of the tick and of the pulls, ns.
    private(set) var tickNs: UInt64 = 0
    private(set) var pullNs: UInt64 = 0

    init(dtMs: Float) {
        self.dtMs = dtMs
        done.enter()
    }

    var isDone: Bool { lock.withLock { finished } }

    /// Block until the queue finished it.
    func wait() { done.wait() }

    /// The tick and its reads, on the preparation queue.
    func prepare(core: TrideScene, signals inbox: SceneSignalInbox, plan: PullPlan?) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        do {
            let tick = try core.tick(dtMs: dtMs)
            result = tick
            let t1 = DispatchTime.now().uptimeNanoseconds
            tickNs = t1 - t0
            // The listeners fired during the tick, on this thread: they are this frame's.
            signals = inbox.drain()
            if let plan, !tick.mutations.isEmpty {
                pulls = CorePulls.prefetch(tick.mutations, plan: plan, core: core)
            }
            pullNs = DispatchTime.now().uptimeNanoseconds - t1
        } catch {
            failure = error
            signals = inbox.drain()
        }
        lock.withLock { finished = true }
        done.leave()
    }
}

#endif
