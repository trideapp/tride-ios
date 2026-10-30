/// Coalesces RFC 6901 JSON-pointer mutations from Rust `TrideScene` into minimal Filament TRS and material updates.
///
/// ## Public API
/// - `PointerDispatcher` — groups `TrideMutation` batches and calls `FilamentBridge` setters once per dirty entity.
///
/// ## Non-obvious constraints
/// - MainActor-only with the Filament bridge. A whole rotation value is the canonical 4-component
///   quaternion `[qx, qy, qz, qw]` (read verbatim into `simd_quatf`); a 3-component Euler XYZ value
///   (radians) from older manifests is still accepted, with a warn-once notice in OSLog.
///   Per-axis `rotation.{x|y|z}` channels are Euler radians; Euler XYZ ↔ quaternion (a node's lanes
///   seeded from its rest pose, light directions) is the core's (`eulerToQuat` / `quatToEuler`), so
///   a pose on a gimbal pole seeds the lanes alike on every host.
/// - A light / camera with a baked carrier node
///   (`listCarrierMetadata`) is posed through that node — spatial channels write the carrier
///   TRS via the index-based node path; the Light / Camera components stay at identity. Its LOCAL
///   pose is held between batches (authored at bind): a lane patches one axis of the held pose, so
///   an axis a batch does not carry keeps its last value while another moves; a complete batch
///   starts from the authored pose again. Its angles turn into a quaternion in the record's own
///   order (a camera's `rotationOrder`) through the core's `eulerToQuat` — never composed here.
///   Cameras are materialised from `listCameras` + `activeCamera`; `/scene/active_camera`
///   switches the View (hard cut).
/// - A batch carries only changed channels, so a value stays until written again. Where
///   the bridge needs a partner value (cone inner / outer, camera projection) it takes the one it
///   last applied; a generator whose `trunkEpoch` did not move gets its transforms, not its mesh.
/// - A complete batch (`complete:` — the core's whole frame after a reset) returns an animated
///   render setting it does not carry (saturation, fog colour, bloom strength, pixelation switch and
///   size) to the scene's own. A pixelation write of the wrong shape (a switch that is not true /
///   false, a size that is not a finite number) is a problem, said once per pointer.
/// - A re-evaluated trunk already on screen uploads on the next tick (`flushDeferredTrunks`), so
///   the core's re-evaluation and the upload do not share a frame; the first mesh is immediate.
/// - A parametric node has no mesh in the GLB — the core builds its datablock's mesh
///   (`datablockMesh`) and the bridge draws it as a generated mesh under the node, in the core's look.
///   A node a deformer bends draws the same way from the core's bent mesh
///   (`deformedMesh`, re-pulled on `/deformed/{uid}/geometry`); an imported one hides its own
///   renderable meanwhile (`setNodeMeshReplaced`) and shows it again when nothing bends it.
/// - Node materials load at bind, before any mesh — each family's package from the
///   archive (on a device, its precompiled Metal build when the app bundles one —
///   `PrecompiledMaterials`), checked against the bridge's Filament first; their uniforms come from the core
///   (`materialGraphValues`), again on every write to the slug — only the uniforms whose lanes moved
///   go to the bridge (`SentUniforms`); a node-graph input (`isGraphLeaf`) never moves
///   the look, so a tick writing only those does not pull it. A missing or foreign package keeps
///   the built-in look. A module batch whose instanced package samples the instance rows gets
///   `generatorModuleColors` with its transforms; no other batch pulls them.
/// - A photo slot shows what the core's groups say (`picture`, else the placeholder),
///   read at registration and again whenever its slug is written; the pixels come from the scene's
///   `HostImageStore`. A picture the store does not hold (a host script's `images` step) keeps the
///   placeholder, said once.
/// - The look the core hands iOS (`materialParams`) carries no glass (`glassFields`): a write to
///   one still recolours the slug and is said once per material and field — an imported glTF glass
///   keeps the value saved in the scene.

import Foundation
import os
import simd

#if canImport(TrideCore)
import TrideCore
#endif

#if canImport(TrideKitFilamentBridge)
import TrideKitFilamentBridge

private let logger = Logger(subsystem: "studio.tride.kit", category: "PointerDispatcher")

/// Signpost events for Instruments (`trunk upload`): free unless a trace records them.
private let signposts = OSSignposter(subsystem: "studio.tride.kit", category: "Performance")

/// One-shot guard for the warn-once notice on a 3-component Euler XYZ
/// rotation value. See `synthesiseQuaternion(from:)`.
private nonisolated(unsafe) var legacyEulerRotationWarned = false

/// Mediates the runtime ↔ renderer boundary. **Not thread-safe** — the
/// caller (`TrideKitScene`) keeps everything pinned to `MainActor`.
@available(iOS 17.0, macCatalyst 17.0, *)
final class PointerDispatcher {

    /// Categorisation of the leading segment of a JSON Pointer. Used
    /// internally and by `PointerDispatcherTests` to keep the parsing
    /// logic verifiable without a Filament instance.
    enum DispatchKind: Equatable {
        case nodeTranslation(index: Int)
        case nodeRotation(index: Int)
        case nodeScale(index: Int)
        case nodeTranslationComponent(index: Int, component: Int)
        case nodeRotationComponent(index: Int, component: Int)
        case nodeScaleComponent(index: Int, component: Int)
        case nodeMorphWeight(nodeIndex: Int, weightIndex: Int)
        /// `/nodes/{i}/visible` (and `/node/{uid}/visible` rewritten) — the node's own flag; what
        /// is drawn is the effective one (AND of every ancestor, like the editor's `effectiveVisible`).
        case nodeVisible(index: Int)
        case materialBaseColor(index: Int)
        /// `/lights/{slug}/<channel>` — manifest-owned light
        /// mutation. `id` is the manifest slug (not an integer index);
        /// `channel` keeps the dispatch data-driven so a
        /// single `.light` arm routes all light properties.
        case light(id: String, channel: LightChannel)
        /// `/cameras/{slug}/<channel>` — manifest-owned camera;
        /// same slug + data-driven channel shape as `.light`.
        case camera(id: String, channel: CameraChannel)
        /// `/scene/active_camera` — hard-cut camera switch.
        case activeCamera
        /// `/environment_settings/<channel>` — manifest-scoped HDR env
        /// authoring (flat 2-segment namespace, NOT under `/scene/`). A single
        /// `.environment` arm routes all animatable env properties,
        /// mirroring the `.light` slug path.
        case environment(channel: EnvironmentChannel)
        /// `/render_settings/<leaf>` — the 3D frame's saturation, the fog colour, the bloom
        /// strength or pixelation's switch and size, flat like the environment (one object, no id).
        case renderSettings(channel: RenderSettingsChannel)
        /// `/generators/{slug}/geometry` — core announcement that
        /// generator `slug` re-evaluated; the buffers are pulled via
        /// `generatedGeometry(id:)`, never carried in the mutation value.
        case generatorGeometry(id: String)
        /// `/generators/{slug}/flow` — the pulse table of generator
        /// `slug` changed; pulled via `generatorFlow(id:)` (`nil` = no pulses left).
        case generatorFlow(id: String)
        /// `/datablocks/{uid}/geometry` — parametric datablock `uid` was
        /// rebuilt at its effective recipe; the mesh is pulled via `datablockMesh(uid:)` for its
        /// nodes and, when one is registered, its module mesh. Recipe leaves are the core's.
        case datablockGeometry(uid: String)
        /// `/deformed/{uid}/geometry` — node `uid` was bent anew by its deformers, or left their
        /// reach; the mesh is pulled via `deformedMesh(nodeUid:)` (`nil` = it draws its own again).
        /// A deformer's leaves (`/deformers/{slug}/…`) are the core's.
        case deformedGeometry(nodeUid: String)
        /// `/materialOverrides/{slug}/<property>` — a timeline / binding write to palette look
        /// `slug`: the resolved look is pulled via `materialParams(slug:)` once per
        /// tick, whatever property moved, and every instance of it recoloured.
        case materialOverride(slug: String)
        case unknown
    }

    /// Property channel of a `/cameras/{slug}/…` pointer.
    /// `position` also accepts the editor grammar's `translation` alias.
    enum CameraChannel: Equatable {
        case position(component: Int?)
        case rotation(component: Int?)
        case yfov
        case znear
        case zfar
        /// The focus plane (m) and the f-number of the depth of field.
        case focusDistance
        case aperture
    }

    /// Spatial lane shared by carrier-anchored lights AND cameras: both write
    /// the same carrier TRS.
    enum SpatialChannel: Equatable {
        case position(component: Int?)
        case rotation(component: Int?)
    }

    /// Animatable channel of an `/environment_settings/…` pointer.
    /// `rotation` is whole-vec3 or per-axis dotted (`rotation.y`); `intensity`
    /// is scalar; `backgroundColor` is the whole linear RGB or one
    /// channel (`background_color.g`, `background_color/1`). `background_visible`
    /// is intentionally absent — not animatable, so it falls through to
    /// `.unknown`.
    enum EnvironmentChannel: Equatable {
        case rotation(component: Int?)
        case intensity
        case backgroundColor(component: Int?)
    }

    /// Animatable render setting: `tone_mapping/saturation` (scalar), `fog/color` (whole linear
    /// RGB, or one channel `fog/color/1`), `bloom/strength` (scalar, over the scene's bloom) and
    /// `pixelation/enabled` (true / false) / `pixelation/size` (block side in points).
    enum RenderSettingsChannel: Equatable {
        case saturation
        case fogColor(component: Int?)
        case bloomStrength
        case pixelationEnabled
        case pixelationSize
    }

    /// Property channel of a `/lights/{slug}/…` pointer. `component == nil`
    /// addresses the whole vector; a non-nil component is a per-axis dotted
    /// pointer (`rotation.x`, `color.g`, …) emitted by the runtime.
    enum LightChannel: Equatable {
        case color(component: Int?)
        case intensity
        case range
        case innerCone
        case outerCone
        case rotation(component: Int?)
        case position(component: Int?)
    }

    private let scene: TrideScene

    /// Material index → list of node indices that should receive opacity
    /// updates when this material's baseColorFactor changes.
    private(set) var materialIndexToNodeIndices: [Int: [Int]] = [:]

    /// Manifest node UID → glTF node index resolver for canonical
    /// `/node/{uid}` mutation pointers. Populated by `bind(to:)`.
    private var nodeIdToGltfIndex: [String: Int] = [:]

    /// Mutation pointer → what it drives. The runtime re-sends the same pointers every tick
    /// (~100 a tick in a typical scene); parsing them again cost ~1 ms of an iPhone 13's frame.
    /// A pointer's kind only depends on `nodeIdToGltfIndex`, so `bind(to:)` starts it afresh.
    private var kindByPointer: [String: DispatchKind] = [:]
    /// Morph weight vectors held for the life of the scene, per node index (a lane absent from a
    /// batch says nothing, so lanes written by different sources survive each other).
    private var heldMorphWeights: [Int: [Float]] = [:]

    /// Manifest light slug → descriptor, cached at `bind(to:)`.
    /// Supplies base values when the runtime emits a partial per-axis
    /// channel (`rotation.x`) so the bridge applies the full vector.
    private var lightDescriptors: [String: TrideLightDescriptor] = [:]

    /// Light / camera slug → glTF node index of its baked carrier node: the one
    /// TransformManager node the entity owns. Spatial channels of a carried entity
    /// are written to that node through the same index-based TRS path mesh nodes
    /// use; a light without a carrier (older archive) keeps the LightManager overlay path.
    private var carrierIndexById: [String: Int] = [:]

    /// Carrier slug → the authored LOCAL pose (`bind(to:)`, from the light / camera descriptor) and
    /// → the pose the batches moved it to: the one pose of the carrier, held between batches.
    private var authoredCarrierPoses: [String: CarrierPose] = [:]
    private(set) var carrierPoses: [String: CarrierPose] = [:]

    /// glTF node index → its rotation as Euler XYZ radians, kept across ticks. A per-axis
    /// `rotation.{x|y|z}` channel is EULER RADIANS (the editor's convention) and may run
    /// past a full turn, so the axes cannot be re-derived from the
    /// quaternion every tick. Seeded once from the node's rest rotation.
    private var nodeEulerByIndex: [Int: SIMD3<Float>] = [:]

    /// Manifest camera slug → descriptor, cached at `bind(to:)`.
    /// Supplies base values for partial per-axis channels and lens fallbacks.
    private var cameraDescriptors: [String: TrideCameraDescriptor] = [:]

    /// warn-once latch: a camera spatial mutation with no carrier to write.
    private var warnedCarrierlessCameras: Set<String> = []
    /// warn-once latch: a carrier whose rotation order the core does not know.
    private var warnedUnknownRotationOrders: Set<String> = []

    /// Authored HDR env base orientation (Euler XYZ radians), cached at
    /// `bind(to:)` from `scene.environment()`. Seeds the missing axes when the
    /// runtime emits a partial per-axis env channel (`rotation.y`) so the
    /// bridge applies the full vector. `.zero` when no env / no settings.
    private var environmentBaseEuler: SIMD3<Float> = .zero

    /// glTF node index → the node's OWN visibility (manifest `visible`, then runtime writes)
    /// and → its parent's index; cached at `bind(to:)`. The drawn visibility is effective —
    /// the node AND every ancestor — applied parents first (`applyNodeVisibility`).
    private var nodeVisibleOwn: [Int: Bool] = [:]
    private var nodeParentIndex: [Int: Int] = [:]

    /// Authored background colour (linear RGB), cached at `bind(to:)` from
    /// `scene.backgroundColor()` — seeds the other channels when the runtime emits one
    /// (`background_color.g`) so the bridge always receives a full colour.
    private var environmentBaseBackground = SIMD3<Float>(0.07, 0.07, 0.07)

    /// The fog colour now (linear RGB), cached at `bind(to:)` from the scene's
    /// normalized render settings and advanced by every fog colour write — seeds the other
    /// channels when the runtime emits one (`fog/color/1`).
    private var renderBaseFogColor = SIMD3<Float>(1, 1, 1)

    /// The scene's own values of the animated render settings, cached at `bind(to:)` — where a
    /// complete batch returns a setting it does not carry. `nil`: the scene authors no render
    /// settings (Filament's defaults stay).
    private var authoredRender: AuthoredRenderSettings?

    init(scene: TrideScene) {
        self.scene = scene
    }

    /// Per-tick accumulation of light-channel mutations for one slug.
    /// Partial per-axis channels (`rotation.x`) seed the missing axes from
    /// the cached descriptor so the bridge always receives a full vector.
    private struct LightAccumulator {
        var color: SIMD3<Float>?
        var intensity: Float?
        var range: Float?
        var innerCone: Float?
        var outerCone: Float?
        var euler: SIMD3<Float>?
        var rotationDirty = false
        var position: SIMD3<Float>?
    }

    /// LOCAL pose of one carrier-anchored light / camera: its position and its record's angles
    /// (radians by axis), patched per axis (multi-turn faithful, no quaternion re-decompose).
    /// The angles turn into the carrier's quaternion in `order` — the record's own — through the
    /// core (`carrierRotation`), never here.
    struct CarrierPose: Equatable {
        var position: SIMD3<Float>
        var euler: SIMD3<Float>
        /// `TrideCameraDescriptor.rotationOrder` for a camera; a light's angles are X Y Z.
        var order: String = PointerDispatcher.lightRotationOrder
    }

    /// The order a light record's angles compose in (`TrideLightDescriptor.rotation`: Euler XYZ).
    static let lightRotationOrder = eulerXYZOrder

    /// The editor's Euler XYZ (`Q = Qx · Qy · Qz`): a node's per-axis rotation lanes and a light
    /// record's angles.
    static let eulerXYZOrder = "xyz"

    /// Per-tick lens overlay of one camera (`/cameras/{id}/yfov|znear|zfar|focusDistance|aperture`).
    private struct CameraLensAccumulator {
        var yfov: Float?
        var znear: Float?
        var zfar: Float?
        var focusDistance: Float?
        var aperture: Float?
    }

    /// Per-tick accumulation of the (single) HDR environment's animatable
    /// channels. Partial per-axis `rotation.y` seeds the missing axes from
    /// `environmentBaseEuler` so the bridge always receives a full Euler vector.
    private struct EnvironmentAccumulator {
        var euler: SIMD3<Float>?
        var rotationDirty = false
        var intensity: Float?
        var backgroundColor: SIMD3<Float>?
    }

    /// Per-tick accumulation of the animatable render settings: the last
    /// write of the tick wins; a fog channel seeds the rest from `renderBaseFogColor`.
    private struct RenderSettingsAccumulator {
        var saturation: Float?
        var fogColor: SIMD3<Float>?
        var bloomStrength: Float?
        var pixelationEnabled: Bool?
        var pixelationSize: Float?
    }

    /// The scene's own values of the animated render settings.
    private struct AuthoredRenderSettings {
        let saturation: Float
        let fogColor: SIMD3<Float>
        let bloomStrength: Float
        let pixelationEnabled: Bool
        let pixelationSize: Float
    }

    /// Build material lookup tables and the node-UID resolver. Transform
    /// dispatch resolves nodes directly through `FilamentBridge`'s
    /// canonical node-index map, built at asset load time from
    /// `getEntities()` + glTF `skins[].joints` + node-name matching
    /// (official iOS Filament has no `getNodeMap()`). Material opacity uses the same
    /// index-based dispatch. Canonical `/node/{uid}` mutation pointers
    /// resolve via `listNodeMetadata()`.
    /// Where problems go besides the log: `TrideKitScene` hands them to the host after the tick.
    var report: (TrideKitDiagnostic) -> Void = { _ in }

    private func problem(
        _ kind: TrideKitDiagnostic.Kind, _ subject: String, _ message: String, code: String? = nil
    ) {
        logger.error("\(kind.rawValue, privacy: .public) \(subject, privacy: .public): \(message, privacy: .public)")
        report(TrideKitDiagnostic(kind: kind, subject: subject, message: message, code: code))
    }

    func bind(to bridge: FilamentBridge) {
        let nodes = readJsonArray(pointer: "/nodes") ?? []
        let meshes = readJsonArray(pointer: "/meshes") ?? []
        materialIndexToNodeIndices = Self.collectMaterialBindingsByIndex(
            fromNodesArray: nodes,
            meshesArray: meshes
        )

        // Node UID → glTF index resolver from manifest metadata.
        nodeIdToGltfIndex = [:]
        kindByPointer = [:]
        heldMorphWeights = [:]
        for entry in scene.listNodeMetadata() {
            nodeIdToGltfIndex[entry.id] = Int(entry.gltfIndex)
        }

        // Carrier table — the baked glTF node that carries each light / camera
        // (a TransformManager child of its group).
        carrierIndexById = [:]
        for entry in scene.listCarrierMetadata() {
            carrierIndexById[entry.id] = Int(entry.gltfIndex)
        }

        // Enumerate manifest-owned lights and materialize them
        // into the bridge (carrier / imported reuse via glTF index, else
        // editor-created build). Cache descriptors for partial-channel bases.
        materializeLights(into: bridge)

        // Manifest cameras reuse the gltfio-born Camera component on
        // their carrier; the authored `scene.activeCamera` seeds the View
        // (runtime switches arrive as `/scene/active_camera` mutations).
        materializeCameras(into: bridge)
        bridge.setActiveCamera(scene.activeCamera())

        // Every carrier starts from its authored pose (a carrier with no descriptor: the origin).
        authoredCarrierPoses = [:]
        for id in carrierIndexById.keys {
            let position = descriptorVec3(lightDescriptors[id]?.position ?? cameraDescriptors[id]?.position)
            let euler = descriptorVec3(lightDescriptors[id]?.rotation ?? cameraDescriptors[id]?.rotation)
            authoredCarrierPoses[id] = CarrierPose(
                position: position ?? .zero,
                euler: euler ?? .zero,
                order: cameraDescriptors[id]?.rotationOrder ?? Self.lightRotationOrder)
        }
        carrierPoses = authoredCarrierPoses

        // Node materials load before any mesh wears them; the asset's slots
        // of their slugs are dressed here, generated meshes and batches pick them up as they come.
        materializeMaterialGraphs(into: bridge)

        // Procedural generators are primed by the core at load —
        // upload their trunk meshes under the hosting node before the first tick.
        materializeGenerators(into: bridge)

        // Parametric nodes — the core's mesh of the recipe under the node.
        materializeParametricNodes(into: bridge)

        // Authored visibility (a hidden template / carrier must not draw): own flags + parents,
        // applied after the generators so what a hidden node hosts is hidden too.
        nodeVisibleOwn = [:]
        nodeParentIndex = [:]
        for entry in scene.listNodeMetadata() {
            let index = Int(entry.gltfIndex)
            nodeVisibleOwn[index] = entry.visible
            if let parent = entry.parentId, let parentIndex = nodeIdToGltfIndex[parent] {
                nodeParentIndex[index] = parentIndex
            }
        }
        applyNodeVisibility(to: bridge)

        // Cache the authored env orientation so per-axis env
        // rotation channels (`rotation.y`) resolve to a full Euler vector.
        if let env = scene.environment(), env.rotation.count >= 3 {
            environmentBaseEuler = SIMD3(
                Float(env.rotation[0]), Float(env.rotation[1]), Float(env.rotation[2])
            )
        } else {
            environmentBaseEuler = .zero
        }
        // The authored fog colour seeds per-channel fog writes; the authored values are where a
        // complete batch returns the animated settings.
        authoredRender = scene.renderSettings().map {
            AuthoredRenderSettings(
                saturation: $0.toneMapping.saturation,
                fogColor: SIMD3($0.fog.colorR, $0.fog.colorG, $0.fog.colorB),
                bloomStrength: $0.bloom.strength,
                pixelationEnabled: $0.pixelation.enabled,
                pixelationSize: $0.pixelation.size
            )
        }
        if let authoredRender { renderBaseFogColor = authoredRender.fogColor }
        // The background is the scene's, environment or not — seed the solid
        // skybox from the manifest (the schema's grey when unauthored) and keep the base
        // for per-channel writes.
        let bg = scene.backgroundColor()
        if bg.count >= 3 {
            environmentBaseBackground = SIMD3(Float(bg[0]), Float(bg[1]), Float(bg[2]))
        }
        bridge.setBackgroundColor(
            r: environmentBaseBackground.x, g: environmentBaseBackground.y, b: environmentBaseBackground.z)

        logger.notice(
            "bound dispatcher: materials=\(self.materialIndexToNodeIndices.count) nodes=\(self.nodeIdToGltfIndex.count) lights=\(self.lightDescriptors.count) cameras=\(self.cameraDescriptors.count) carriers=\(self.carrierIndexById.count) activeCamera=\(self.scene.activeCamera() ?? "-", privacy: .public)"
        )
    }

    /// Enumerate `scene.listLights()`, cache descriptors keyed by slug, and
    /// hand the bridge plain ObjC dictionaries (Euler `rotation` pre-resolved
    /// to a `-Z` emission `direction` here) so it can
    /// reuse gltfio-born Light components or build editor-created ones.
    private func materializeLights(into bridge: FilamentBridge) {
        let lights = scene.listLights()
        lightDescriptors = Dictionary(
            lights.map { ($0.id, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        let dicts = lights.map {
            Self.lightDictionary(from: $0, carrierGltfIndex: carrierIndexById[$0.id])
        }
        bridge.addLights(dicts)
    }

    // Generator slug → hosting node glTF index / palette material.
    private var generatorHostIndex: [String: Int] = [:]
    private var generatorMaterialSlug: [String: String] = [:]
    /// Palette entry behind the slug — applied by the bridge when the baked asset
    /// carries no instance for it.
    private var generatorMaterialParams: [String: TrideMaterialParams] = [:]
    /// Pulse look per generator whose recipe carries a `pulse` block.
    private var generatorPulseLook: [String: TridePulseLook] = [:]
    /// Module bindings per generator: symbol → datablock, instanced per frame.
    private var generatorModules: [String: [TrideModuleBindingDescriptor]] = [:]
    /// Datablock uid → content hash of the decoded mesh the bridge holds (decoded once per
    /// scene; the bridge swaps buffers when a uid arrives with another hash).
    private var registeredModuleMeshes: [String: String] = [:]
    /// Datablock uid → the draws its module mesh makes: one part per draw group of an
    /// Extrude cloned as a module — a derived uid, the group's slice of the indices, its own look —
    /// or the mesh itself. Every per-datablock cache below is keyed by the PART's uid.
    private var moduleParts: [String: [ModulePart]] = [:]
    /// Palette entry of each module datablock's material (resolved by the core with the mesh).
    private var moduleMaterialParams: [String: TrideMaterialParams?] = [:]
    /// Palette slug of each module datablock's material — the key a recolour finds it by.
    private var moduleMaterialSlug: [String: String] = [:]
    /// The look last handed to the bridge per palette slug — an unchanged one is
    /// not re-sent (a scrubbed clip writes every tick).
    private var paletteSent: [String: TrideMaterialParams] = [:]
    /// Archive bytes behind palette `baseColorTexture` uris, read once per scene.
    private var assetBytesCache: [String: Data] = [:]
    /// The trunk epoch each generator's mesh was last uploaded at: an announcement
    /// with the same `trunkEpoch` moves only the module / instance transforms — a cloner's is
    /// always 0, so its empty trunk is uploaded once.
    private var uploadedTrunkEpoch: [String: UInt64] = [:]
    /// Last cone / projection handed to the bridge per light / camera. A tick carries only the
    /// channels that changed, so the partner of a changed one comes from here, not
    /// from the authored descriptor.
    private var liveCone: [String: (inner: Float, outer: Float)] = [:]
    private var liveProjection: [String: (yfov: Float, znear: Float, zfar: Float)] = [:]
    /// The look dictionary handed to a datablock's batches and its `lookRevision`: rebuilt only
    /// when the look changes, so a sync moves instances without re-applying an unchanged look.
    private var moduleLook: [String: (params: TrideMaterialParams, slug: String?, dict: [String: Any])] = [:]
    /// Re-evaluated trunks whose new mesh waits for the next tick (`flushDeferredTrunks`), with
    /// the trunk epoch they will hold.
    private var deferredTrunks: [(id: String, trunk: UInt64?)] = []
    private var nextLookRevision: Int64 = 0
    /// Parametric nodes of the scene, listed once at attach.
    private var parametricNodes: [TrideParametricNode] = []
    /// Imported nodes drawn bent now: their own renderable gives way to the bent mesh.
    private var bentImported: Set<String> = []

    /// Generator id → the uid of the node that hosts it (a node material's object).
    private var generatorHostUid: [String: String] = [:]
    /// Node material slug → the families its packages loaded for on the bridge.
    private var materialGraphFamilies: [String: [String]] = [:]
    /// The uniforms last handed to the bridge per node material and family.
    private var sentUniforms = SentUniforms()
    /// Node material slugs whose instanced package loaded and samples the instance rows
    /// (`TrideMaterialGraph.instanceColors`) — their module batches pull them.
    private var instanceRowSlugs: Set<String> = []

    /// What a photo slot's group binds.
    enum SlotBinding: Equatable {
        case placeholder
        case picture(HostImageKey)
    }
    /// The app's pictures (`TrideKitScene.setImage`) the slots read their pixels from.
    var hostImages: HostImageStore?
    /// What each photo slot shows on the bridge, by `slug\0family\0sampler`.
    private var slotBindings: [String: SlotBinding] = [:]
    /// Node material slugs with a photo slot on a loaded package — their groups are read again
    /// whenever the slug is written.
    private var slotSlugs: Set<String> = []
    /// Pictures the core shows that the store never held — said once each.
    private var missingPictures: Set<HostImageKey> = []

    /// The pictures some slot shows now — the store keeps them.
    var boundHostPictures: Set<HostImageKey> {
        Set(slotBindings.values.compactMap { binding in
            if case .picture(let key) = binding { return key }
            return nil
        })
    }

    /// What each photo slot of `groups` should show: the app's picture the core names, else the
    /// placeholder. Groups without an image input are not slots.
    static func slotTargets(_ groups: [TrideMaterialGraphGroup]) -> [(sampler: String, binding: SlotBinding)] {
        groups.compactMap { group in
            guard let name = group.hostImage else { return nil }
            guard let picture = group.picture else { return (group.sampler, .placeholder) }
            return (group.sampler, .picture(HostImageKey(name: name, revision: picture.revision)))
        }
    }

    /// Bring the photo slots of `slug` on `family` to what the core's groups say.
    private func syncSlots(
        slug: String, family: String, groups: [TrideMaterialGraphGroup], into bridge: FilamentBridge
    ) {
        for target in Self.slotTargets(groups) {
            let key = "\(slug)\0\(family)\0\(target.sampler)"
            guard slotBindings[key] != target.binding else { continue }
            var binding = target.binding
            if case .picture(let picture) = target.binding {
                if let held = hostImages?.picture(for: picture) {
                    if bridge.bindMaterialGraphSlot(
                        slug, family: family, sampler: target.sampler, picture: picture.name,
                        revision: picture.revision, width: held.width, height: held.height, pixels: held.pixels)
                    {
                        slotBindings[key] = binding
                        continue
                    }
                    // The core takes the failure: the input turns `failed`, the slots go back to
                    // their placeholders with the next step.
                    let reason = "the view could not upload the picture (\(held.width) × \(held.height))"
                    problem(.load, picture.name, reason)
                    try? scene.imageFailed(name: picture.name, reason: reason)
                } else if missingPictures.insert(picture).inserted {
                    problem(.load, picture.description, "the scene shows a picture this host never received — the placeholder stays")
                }
                binding = .placeholder
            }
            guard slotBindings[key] != binding else { continue }
            if bridge.bindMaterialGraphSlot(
                slug, family: family, sampler: target.sampler, picture: nil, revision: 0, width: 0, height: 0,
                pixels: nil)
            {
                slotBindings[key] = binding
            }
        }
    }

    /// Load every node material's packages — checked against the bridge's
    /// own Filament first (a foreign package aborts in `Material::Builder`) — with their texture
    /// arrays and current uniforms, then dress the asset's slots that wear them. A missing or
    /// refused package leaves the slug on the built-in path; the core's problems
    /// (`materialGraphProblems`: stale / version records and a package the archive does not hold,
    /// `material.package-missing`) are told once per material and code, in its words.
    private func materializeMaterialGraphs(into bridge: FilamentBridge) {
        materialGraphFamilies = [:]
        sentUniforms.removeAll()
        instanceRowSlugs = []
        slotBindings = [:]
        slotSlugs = []
        var toldRecords = Set<String>()
        for told in scene.materialGraphProblems() where toldRecords.insert("\(told.slug)\u{0}\(told.code)").inserted {
            problem(.load, told.slug, told.message, code: told.code)
        }
        let slugs = scene.materialGraphSlugs()
        guard !slugs.isEmpty else { return }
        let engineVersion = bridge.referenceMaterialPackage.flatMap { materialPackageInfo(bytes: $0) }?.version
        var precompiled = 0
        for slug in slugs {
            for family in ["mesh", "generated", "instanced"] {
                guard let graph = scene.materialGraph(slug: slug, family: family) else { continue }
                // A package the archive does not hold is the core's problem, told above.
                guard let shipped = scene.assetBytes(uri: graph.packageUri) else { continue }
                var package = shipped
                do {
                    if let built = try PrecompiledMaterials.package(for: shipped) {
                        package = built
                        precompiled += 1
                    }
                } catch {
                    problem(.load, slug, "node material (\(family)): precompiled Metal library unreadable (\(error.localizedDescription)) — compiled on the device")
                }
                guard let info = materialPackageInfo(bytes: package), info.version == engineVersion,
                      info.backends.contains("metal")
                else {
                    problem(.load, slug, "node material (\(family)): package for another Filament or without Metal — built-in look")
                    continue
                }
                let arrays: [[String: Any]] = graph.groups.map { group in
                    var array: [String: Any] = [
                        "sampler": group.sampler,
                        "srgb": NSNumber(value: group.srgb),
                        "width": NSNumber(value: group.width),
                        "height": NSNumber(value: group.height),
                        "layers": group.layers.map { ref -> Any in
                            guard let ref, let bytes = cachedAssetBytes(ref.uri) else { return NSNull() }
                            return Self.textureRefDictionary(ref, bytes: bytes)
                        },
                    ]
                    // A photo slot: its placeholder (or one white layer) until the app's picture.
                    if let name = group.hostImage { array["hostImage"] = name }
                    return array
                }
                let params = graph.params.map { ["name": $0.name, "kind": $0.kind] }
                guard bridge.registerMaterialGraph(
                    slug, family: family, package: package, arrays: arrays, params: params,
                    instanceColors: graph.instanceColors, castShadows: graph.castShadows)
                else { continue }
                materialGraphFamilies[slug, default: []].append(family)
                if family == "instanced" && graph.instanceColors { instanceRowSlugs.insert(slug) }
                pushMaterialGraphValues(slug: slug, family: family, into: bridge)
                let slots = Self.slotTargets(graph.groups)
                if !slots.isEmpty {
                    slotSlugs.insert(slug)
                    // Registration bound each slot's placeholder; a picture already shown follows.
                    for slot in slots { slotBindings["\(slug)\0\(family)\0\(slot.sampler)"] = .placeholder }
                    syncSlots(slug: slug, family: family, groups: graph.groups, into: bridge)
                }
            }
        }
        var seeds: [NSNumber: NSNumber] = [:]
        for (uid, index) in nodeIdToGltfIndex {
            seeds[NSNumber(value: index)] = NSNumber(value: materialObjectSeed(uid: uid))
        }
        let dressed = bridge.dressAssetWithMaterialGraphs(objectSeeds: seeds)
        logger.notice(
            "node materials: \(self.materialGraphFamilies.count) of \(slugs.count) loaded (\(precompiled) package(s) precompiled), \(dressed) asset slot(s) dressed")
    }

    /// The core's uniforms of `slug` on `family` at the scene's current values that moved since the
    /// last push, to every instance; nothing moved — no call.
    private func pushMaterialGraphValues(slug: String, family: String, into bridge: FilamentBridge) {
        let moved = sentUniforms.changes(
            slug: slug, family: family, values: scene.materialGraphValues(slug: slug, family: family))
        guard !moved.isEmpty else { return }
        var values: [String: [NSNumber]] = [:]
        for value in moved {
            values[value.name] = value.values.map { NSNumber(value: $0) }
        }
        _ = bridge.updateMaterialGraph(slug, family: family, values: values)
    }

    private func materializeGenerators(into bridge: FilamentBridge) {
        generatorHostIndex = [:]
        generatorHostUid = [:]
        generatorMaterialSlug = [:]
        generatorMaterialParams = [:]
        generatorPulseLook = [:]
        generatorModules = [:]
        registeredModuleMeshes = [:]
        moduleParts = [:]
        moduleMaterialParams = [:]
        moduleMaterialSlug = [:]
        paletteSent = [:]
        assetBytesCache = [:]
        uploadedTrunkEpoch = [:]
        moduleLook = [:]
        deferredTrunks = []
        let generators = scene.listGenerators()
        for desc in generators {
            if let uid = desc.nodeUid, let idx = nodeIdToGltfIndex[uid] {
                generatorHostIndex[desc.id] = idx
                generatorHostUid[desc.id] = uid
            }
            if let material = desc.material {
                generatorMaterialSlug[desc.id] = material
            }
            if let params = desc.materialParams {
                generatorMaterialParams[desc.id] = params
            }
            if let pulse = desc.pulse {
                generatorPulseLook[desc.id] = pulse
            }
            if !desc.modules.isEmpty {
                generatorModules[desc.id] = desc.modules
            }
            uploadGeneratedMesh(id: desc.id, into: bridge)
            uploadedTrunkEpoch[desc.id] = desc.kind == "cloner" ? 0 : desc.epoch
        }
        if !generators.isEmpty {
            logger.notice("materialized \(generators.count) generator mesh(es)")
        }
    }

    /// Every parametric node draws its datablock's mesh — built by the core
    /// from the recipe — through the generated-mesh path, under the node's own entity, in the
    /// core's look (the palette entry, else the core's default look, white plastic).
    private func materializeParametricNodes(into bridge: FilamentBridge) {
        parametricNodes = scene.listParametricNodes()
        bentImported = []
        for node in parametricNodes {
            uploadParametricNode(node, into: bridge)
        }
        if !parametricNodes.isEmpty {
            logger.notice("materialized \(self.parametricNodes.count) parametric node(s)")
        }
        // The nodes the core bent at load (a parametric one drew bent above).
        let bent = scene.listDeformedNodes()
        for uid in bent { refreshDeformed(node: uid, into: bridge) }
        if !bent.isEmpty {
            logger.notice("materialized \(bent.count) bent node(s)")
        }
    }

    /// Node `uid` was bent anew, or left every deformer's reach. A parametric node
    /// draws its bent mesh, else its recipe's; an imported one draws the bent mesh under it with its
    /// own renderable off, and takes its renderable back when nothing bends it.
    private func refreshDeformed(node uid: String, into bridge: FilamentBridge) {
        if let node = parametricNodes.first(where: { $0.nodeUid == uid }) {
            uploadParametricNode(node, into: bridge)
            return
        }
        guard let parent = nodeIdToGltfIndex[uid] else {
            problem(.load, uid, "bent node has no glTF node — not drawn")
            return
        }
        let id = "parametric:\(uid)"
        guard let mesh = scene.deformedMesh(nodeUid: uid) else {
            guard bentImported.remove(uid) != nil else { return }
            bridge.removeGeneratedMesh(id)
            bridge.setNodeMeshReplaced(atIndex: parent, replaced: false)
            return
        }
        guard upload(mesh: mesh, id: id, parent: parent, objectUid: uid, into: bridge) else {
            problem(.load, id, "the view rejected the bent mesh")
            return
        }
        if bentImported.insert(uid).inserted {
            bridge.setNodeMeshReplaced(atIndex: parent, replaced: true)
        }
    }

    /// Datablock `uid` was rebuilt at its effective recipe — its nodes
    /// draw the new mesh, and a module mesh registered under it (clones, L-system modules) is
    /// registered again (the bridge moves its batches to the new buffers).
    private func refreshParametric(datablock uid: String, into bridge: FilamentBridge) {
        for node in parametricNodes where node.datablockUid == uid {
            uploadParametricNode(node, into: bridge)
        }
        guard moduleParts[uid] != nil, let mesh = scene.datablockMesh(uid: uid) else { return }
        let before = moduleParts[uid] ?? []
        guard let parts = registerModuleMesh(datablock: uid, mesh: mesh, into: bridge) else {
            problem(.load, uid, "the view rejected the rebuilt module mesh")
            return
        }
        // A group that went (a fillet switched off): its batches hide.
        let kept = Set(parts.map(\.uid))
        for gone in before where !kept.contains(gone.uid) {
            for (id, modules) in generatorModules {
                for binding in modules where binding.datablock == uid {
                    _ = bridge.syncModuleInstances(
                        id, symbol: binding.symbol + gone.suffix, meshUid: gone.uid, transforms: Data(),
                        colors: nil, parentGltfIndex: nil, materialParams: nil)
                }
            }
        }
    }

    /// One draw of a module mesh (`moduleParts`).
    private struct ModulePart {
        /// `""` for a mesh without groups, else `#<group index>`: appended to the batch's symbol.
        let suffix: String
        /// The datablock's uid, or `<uid>#<i>` for a group.
        let uid: String
        let params: TrideMaterialParams?
        let slug: String?
    }

    /// Register a module mesh with the bridge — whole, or one sub-mesh per draw group (the whole
    /// vertex streams, the group's slice of the indices, content key `<hash>#<i>`) — and cache each
    /// part's look. `nil` when the bridge rejects one.
    private func registerModuleMesh(
        datablock uid: String, mesh: TrideDatablockMesh, into bridge: FilamentBridge
    ) -> [ModulePart]? {
        var parts: [ModulePart] = []
        if mesh.groups.isEmpty {
            guard bridge.registerModuleMesh(uid, mesh: Self.moduleMeshDictionary(from: mesh)) else { return nil }
            registeredModuleMeshes[uid] = mesh.contentHash
            parts.append(ModulePart(suffix: "", uid: uid, params: mesh.materialParams, slug: mesh.material))
        } else {
            for (i, group) in mesh.groups.enumerated() {
                let part = ModulePart(
                    suffix: "#\(i)", uid: "\(uid)#\(i)", params: group.materialParams ?? mesh.materialParams,
                    slug: group.material)
                var dict = Self.moduleMeshDictionary(from: mesh)
                // Indices are little-endian u32: the group's slice, whole vertex streams.
                dict["indices"] = mesh.indices.subdata(
                    in: Int(group.indexStart) * 4..<(Int(group.indexStart) + Int(group.indexCount)) * 4)
                dict["contentHash"] = "\(mesh.contentHash)\(part.suffix)"
                guard bridge.registerModuleMesh(part.uid, mesh: dict) else { return nil }
                registeredModuleMeshes[part.uid] = "\(mesh.contentHash)\(part.suffix)"
                parts.append(part)
            }
        }
        for part in parts {
            moduleMaterialParams[part.uid] = part.params
            if let slug = part.slug { moduleMaterialSlug[part.uid] = slug } else { moduleMaterialSlug[part.uid] = nil }
        }
        moduleParts[uid] = parts
        return parts
    }

    private func uploadParametricNode(_ node: TrideParametricNode, into bridge: FilamentBridge) {
        guard let parent = nodeIdToGltfIndex[node.nodeUid] else {
            problem(.load, node.nodeUid, "parametric node has no glTF node — not drawn")
            return
        }
        // A node a deformer reaches draws its bent mesh, else its recipe's.
        guard let mesh = scene.deformedMesh(nodeUid: node.nodeUid) ?? scene.datablockMesh(uid: node.datablockUid) else {
            problem(.load, node.nodeUid, "parametric datablock \(node.datablockUid) could not be built — not drawn")
            return
        }
        let id = "parametric:\(node.nodeUid)"
        if !upload(mesh: mesh, id: id, parent: parent, objectUid: node.nodeUid, into: bridge) {
            problem(.load, id, "the view rejected the parametric mesh")
        }
    }

    /// One core mesh drawn as generated mesh `id` under glTF node `parent`, in the mesh's look.
    private func upload(
        mesh: TrideDatablockMesh, id: String, parent: Int, objectUid: String, into bridge: FilamentBridge
    ) -> Bool {
        let geometry = TrideGeneratedGeometry(
            id: id, epoch: 0, topology: "triangles", positions: mesh.positions,
            normals: mesh.normals, uvs: mesh.uvs, flow: Data(), colors: mesh.colors,
            // A multi-material import keeps a draw group per primitive, each in its look.
            indices: mesh.indices, groups: mesh.groups, tangents: mesh.tangents)
        let dict = Self.geometryDictionary(
            from: geometry,
            parentGltfIndex: parent,
            materialSlug: mesh.material,
            materialParams: mesh.materialParams,
            objectUid: objectUid,
            assetBytes: cachedAssetBytes
        )
        return bridge.upsertGeneratedMesh(id, geometry: dict)
    }

    /// Draw the datablock-bound module symbols of `id` as
    /// instanced batches — the mesh is decoded by the core once per datablock (`datablockMesh`),
    /// the per-instance `mat4[]` comes packed from the core (`generatorModuleTransforms`), and
    /// the instance rows (`generatorModuleColors`) where the batch's node material samples them.
    private func syncModuleInstances(id: String, into bridge: FilamentBridge) {
        guard let modules = generatorModules[id], !modules.isEmpty else { return }
        for binding in modules {
            if moduleParts[binding.datablock] == nil {
                guard let mesh = scene.datablockMesh(uid: binding.datablock) else {
                    problem(.load, id, "module '\(binding.symbol)': datablock \(binding.datablock) could not be decoded — not drawn")
                    continue
                }
                guard registerModuleMesh(datablock: binding.datablock, mesh: mesh, into: bridge) != nil else {
                    problem(.load, binding.datablock, "the view rejected the module mesh")
                    continue
                }
            }
            let transforms = scene.generatorModuleTransforms(id: id, symbol: binding.symbol)
            // One batch per draw part over the same transforms and rows (`moduleParts`).
            for part in moduleParts[binding.datablock] ?? [] {
                let params = moduleMaterialParams[part.uid] ?? nil
                // The batch wears its slug's node material only through a look (`materialParams`).
                let pullsRows = params != nil
                    && moduleMaterialSlug[part.uid].map(instanceRowSlugs.contains) == true
                let placed = bridge.syncModuleInstances(
                    id,
                    symbol: binding.symbol + part.suffix,
                    meshUid: part.uid,
                    transforms: transforms,
                    colors: pullsRows ? scene.generatorModuleColors(id: id, symbol: binding.symbol) : nil,
                    parentGltfIndex: generatorHostIndex[id].map { NSNumber(value: $0) },
                    materialParams: params.map { moduleLookDictionary(datablock: part.uid, params: $0) }
                )
                logger.debug(
                    "generator '\(id, privacy: .public)' module '\(binding.symbol + part.suffix, privacy: .public)': \(placed)/\(transforms.count / 64) instances")
            }
        }
    }

    /// What `updatePaletteMaterial` does not re-apply: the maps and the alpha cut.
    private static func sameMapsAndCut(_ a: TrideMaterialParams, _ b: TrideMaterialParams) -> Bool {
        a.baseColorTexture == b.baseColorTexture && a.emissiveTexture == b.emissiveTexture
            && a.alphaMode == b.alphaMode && a.alphaCutoff == b.alphaCutoff
    }

    private func moduleLookDictionary(datablock: String, params: TrideMaterialParams) -> [String: Any] {
        let slug = moduleMaterialSlug[datablock]
        if let cached = moduleLook[datablock], cached.params == params, cached.slug == slug {
            return cached.dict
        }
        nextLookRevision += 1
        var dict = Self.materialParamsDictionary(params, slug: slug, assetBytes: cachedAssetBytes)
        dict["lookRevision"] = NSNumber(value: nextLookRevision)
        moduleLook[datablock] = (params, slug, dict)
        return dict
    }

    /// Pull palette look `slug` as the scene draws it now and recolour every instance
    /// of it; the caches later uploads read from take the new look too, so a re-grown trunk or a
    /// re-synced batch never falls back to the authored colours. `lookWritten` = the tick wrote a
    /// leaf other than a node-graph input: without it the look cannot have moved and is not pulled.
    private func recolourPalette(slug: String, lookWritten: Bool, into bridge: FilamentBridge) {
        // A node material's uniforms come from the core's table; its photo slots from its groups.
        for family in materialGraphFamilies[slug] ?? [] {
            pushMaterialGraphValues(slug: slug, family: family, into: bridge)
            if slotSlugs.contains(slug) {
                syncSlots(
                    slug: slug, family: family,
                    groups: scene.materialGraph(slug: slug, family: family)?.groups ?? [], into: bridge)
            }
        }
        guard lookWritten || paletteSent[slug] == nil else { return }
        guard let params = scene.materialParams(slug: slug) else {
            warnUnknownPaletteOnce(slug)
            return
        }
        if paletteSent[slug] == params { return }
        paletteSent[slug] = params
        for (id, material) in generatorMaterialSlug where material == slug {
            generatorMaterialParams[id] = params
        }
        for (datablock, material) in moduleMaterialSlug where material == slug {
            moduleMaterialParams[datablock] = params
            // The recolour below carries the factors to the batches; with the same maps and cut
            // their look keeps its revision, so the next sync moves instances only.
            if let cached = moduleLook[datablock], Self.sameMapsAndCut(cached.params, params) {
                var dict = Self.materialParamsDictionary(params, slug: cached.slug, assetBytes: cachedAssetBytes)
                dict["lookRevision"] = cached.dict["lookRevision"]
                moduleLook[datablock] = (params, cached.slug, dict)
            }
        }
        for (id, look) in generatorPulseLook where look.material == slug {
            generatorPulseLook[id] = TridePulseLook(material: slug, materialParams: params)
        }
        let touched = bridge.updatePaletteMaterial(
            slug, params: Self.materialParamsDictionary(params, slug: slug))
        logger.debug("palette '\(slug, privacy: .public)': \(touched) instance(s) recoloured")
    }

    /// Look fields of glTF glass (transmission, IOR, volume, dispersion) the look this kit
    /// recolours from does not carry: the web animates them, iOS keeps the saved value.
    static let glassFields: Set<String> = [
        "transmission", "ior", "thickness", "attenuationColor", "attenuationDistance", "dispersion",
    ]

    /// The glass field a `/materialOverrides/{slug}/{field}[/{lane} | .{lane}]` pointer writes.
    static func glassField(of pointer: String) -> String? {
        let segs = pointer.split(separator: "/", omittingEmptySubsequences: false)
        guard segs.count >= 4, segs[1] == "materialOverrides" else { return nil }
        let field = String(segs[3].prefix { $0.isLetter })
        return glassFields.contains(field) ? field : nil
    }

    /// Whether `pointer` writes a node-graph input of a palette material
    /// (`/materialOverrides/{slug}/nodes/{node}/inputs/{input}[/…]`) — the rule the
    /// web's `isMaterialGraphLeaf` and Android's `isGraphLeaf` follow.
    static func isGraphLeaf(_ pointer: String) -> Bool {
        let segs = pointer.split(separator: "/", omittingEmptySubsequences: false)
        return segs.count >= 7 && segs[1] == "materialOverrides" && segs[3] == "nodes" && !segs[4].isEmpty
            && segs[5] == "inputs" && !segs[6].isEmpty
    }

    private var glassWarned: Set<String> = []
    private func warnGlassOnce(slug: String, field: String) {
        let subject = "/materialOverrides/\(slug)/\(field)"
        guard glassWarned.insert(subject).inserted else { return }
        problem(
            .pointer, subject,
            "material '\(slug)': its glass \(field) does not animate on iOS yet — it keeps the value saved in the scene")
    }

    private var unknownPaletteWarned: Set<String> = []
    /// Pointers already reported as unknown (`dispatch`).
    private var unknownPointers: Set<String> = []

    /// A view-domain write of a value the view cannot draw — said once per pointer (R-NoSilentDrop).
    private func valueShapeProblem(_ mutation: TrideMutation, _ expected: String) {
        guard unknownPointers.insert(mutation.pointer).inserted else { return }
        problem(.pointer, mutation.pointer, "\(expected), got \(mutation.valueJson) — ignored")
    }
    private func warnUnknownPaletteOnce(_ slug: String) {
        guard unknownPaletteWarned.insert(slug).inserted else { return }
        problem(.pointer, slug, "palette material is animated but not in the scene — write ignored")
    }

    private func cachedAssetBytes(_ uri: String) -> Data? {
        if let hit = assetBytesCache[uri] { return hit }
        guard let bytes = scene.assetBytes(uri: uri) else { return nil }
        assetBytesCache[uri] = bytes
        return bytes
    }

    /// Bridge dictionary for a palette entry; `assetBytes` supplies the image behind a
    /// `baseColorTexture` — absent bytes leave the map out (factor only).
    static func materialParamsDictionary(
        _ p: TrideMaterialParams,
        slug: String? = nil,
        assetBytes: (String) -> Data? = { _ in nil }
    ) -> [String: Any] {
        var dict: [String: Any] = [
            "baseColor": p.baseColor.map { NSNumber(value: $0) },
            "metallic": NSNumber(value: p.metallic),
            "roughness": NSNumber(value: p.roughness),
            "emissive": p.emissive.map { NSNumber(value: $0) },
        ]
        // The look's slug rides along so the bridge can recolour its instances.
        if let slug { dict["slug"] = slug }
        // `mask` cuts at `alphaCutoff` (a leaf texture's alpha shapes the blade).
        if let mode = p.alphaMode { dict["alphaMode"] = mode }
        if let cutoff = p.alphaCutoff { dict["alphaCutoff"] = NSNumber(value: cutoff) }
        // The thin-surface look — the bridge picks the family from these.
        if let dt = p.diffuseTransmission { dict["diffuseTransmission"] = NSNumber(value: dt) }
        if let c = p.diffuseTransmissionColor {
            dict["diffuseTransmissionColor"] = c.map { NSNumber(value: $0) }
        }
        if let color = p.rimColor, let power = p.rimPower {
            dict["rim"] = ["color": color.map { NSNumber(value: $0) }, "power": NSNumber(value: power)]
        }
        // Base-colour map and emissive glow mask: same ref shape; a map rides only when its
        // bytes resolve (factor only otherwise). The thin look's transmission colour map rides
        // the same way.
        for (key, texture) in [
            ("baseColorTexture", p.baseColorTexture),
            ("emissiveTexture", p.emissiveTexture),
            ("diffuseTransmissionColorTexture", p.diffuseTransmissionColorTexture),
        ] {
            if let t = texture, let bytes = assetBytes(t.uri) {
                dict[key] = textureRefDictionary(t, bytes: bytes)
            }
        }
        return dict
    }

    /// Bridge dictionary of one resolved palette texture (uri, bytes, glTF sampler enums).
    private static func textureRefDictionary(_ t: TrideTextureRef, bytes: Data) -> [String: Any] {
        var ref: [String: Any] = ["uri": t.uri, "bytes": bytes, "texCoord": NSNumber(value: t.texCoord)]
        if let v = t.mimeType { ref["mimeType"] = v }
        if let v = t.magFilter { ref["magFilter"] = NSNumber(value: v) }
        if let v = t.minFilter { ref["minFilter"] = NSNumber(value: v) }
        if let v = t.wrapS { ref["wrapS"] = NSNumber(value: v) }
        if let v = t.wrapT { ref["wrapT"] = NSNumber(value: v) }
        return ref
    }

    /// Pull the core buffers for `id` and upload them (create or update in place).
    private func uploadGeneratedMesh(id: String, into bridge: FilamentBridge) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        guard let geometry = scene.generatedGeometry(id: id) else {
            problem(.load, id, "generator announced but has no geometry (its evaluation failed)")
            return
        }
        let t1 = DispatchTime.now().uptimeNanoseconds
        let dict = Self.geometryDictionary(
            from: geometry,
            parentGltfIndex: generatorHostIndex[id],
            materialSlug: generatorMaterialSlug[id],
            materialParams: generatorMaterialParams[id],
            pulse: generatorPulseLook[id],
            objectUid: generatorHostUid[id],
            assetBytes: cachedAssetBytes
        )
        let t2 = DispatchTime.now().uptimeNanoseconds
        if !bridge.upsertGeneratedMesh(id, geometry: dict) {
            problem(.load, id, "the view rejected the generated mesh (epoch \(geometry.epoch))")
        }
        let t3 = DispatchTime.now().uptimeNanoseconds
        // Perf signal: FFI lift / dictionary copy / Filament upload per re-evaluation.
        logger.debug(
            "generated mesh '\(id, privacy: .public)' verts=\(geometry.positions.count / 12) ffi=\(Double(t1 - t0) / 1e6, format: .fixed(precision: 2))ms dict=\(Double(t2 - t1) / 1e6, format: .fixed(precision: 2))ms upload=\(Double(t3 - t2) / 1e6, format: .fixed(precision: 2))ms"
        )
        syncModuleInstances(id: id, into: bridge)
    }

    /// A decoded datablock as the bridge registers a module mesh: the core's streams as raw LE
    /// bytes; `colors` and `tangents` empty → the bridge's white slot and
    /// frames from the normals.
    static func moduleMeshDictionary(from mesh: TrideDatablockMesh) -> [String: Any] {
        [
            "positions": mesh.positions,
            "normals": mesh.normals,
            "uvs": mesh.uvs,
            "colors": mesh.colors,
            "tangents": mesh.tangents,
            "indices": mesh.indices,
            "contentHash": mesh.contentHash,
        ]
    }

    static func geometryDictionary(
        from geometry: TrideGeneratedGeometry,
        parentGltfIndex: Int?,
        materialSlug: String?,
        materialParams: TrideMaterialParams? = nil,
        pulse: TridePulseLook? = nil,
        objectUid: String? = nil,
        assetBytes: (String) -> Data? = { _ in nil }
    ) -> [String: Any] {
        var dict: [String: Any] = [
            "topology": geometry.topology,
            "epoch": NSNumber(value: geometry.epoch),
            // Raw little-endian f32/u32 byte streams straight from the FFI (`Data`, memcpy) —
            // the bridge consumes them as-is.
            "positions": geometry.positions,
            "normals": geometry.normals,
            "uvs": geometry.uvs,
            "indices": geometry.indices,
            // The core's MikkTSpace frame of a normal-mapped mesh; empty otherwise.
            "tangents": geometry.tangents,
        ]
        if let parentGltfIndex { dict["parentGltfIndex"] = NSNumber(value: parentGltfIndex) }
        if let materialSlug { dict["materialSlug"] = materialSlug }
        // The scene node the mesh is — a node material's Random Color per object.
        if let objectUid {
            dict["objectUid"] = objectUid
            dict["objectSeed"] = NSNumber(value: materialObjectSeed(uid: objectUid))
        }
        if let p = materialParams {
            dict["materialParams"] = materialParamsDictionary(p, assetBytes: assetBytes)
        }
        // One primitive per pen over the shared streams, each with its palette look.
        if !geometry.groups.isEmpty {
            dict["groups"] = geometry.groups.map { group -> [String: Any] in
                var entry: [String: Any] = [
                    "indexStart": NSNumber(value: group.indexStart),
                    "indexCount": NSNumber(value: group.indexCount),
                ]
                if let slug = group.material { entry["materialSlug"] = slug }
                if let p = group.materialParams {
                    entry["materialParams"] = materialParamsDictionary(p, assetBytes: assetBytes)
                }
                return entry
            }
        }
        // The trunk's colour ramp (linear RGBA f32 per vertex) rides the COLOR slot.
        if !geometry.colors.isEmpty {
            dict["colors"] = geometry.colors
        }
        // A pulsed trunk carries the flow stream + the look the pulse mask blends in.
        if let pulse, !geometry.flow.isEmpty {
            dict["flow"] = geometry.flow
            var look: [String: Any] = [:]
            if let p = pulse.materialParams {
                look["materialParams"] = materialParamsDictionary(p, assetBytes: assetBytes)
            }
            dict["pulse"] = look
        }
        return dict
    }

    /// Bridge dictionary of the core's pulse table (`setGeneratedMeshFlow`), rows as is.
    static func pulseTableDictionary(from table: TridePulseTable) -> [String: Any] {
        [
            "rows": table.rows,
            "capacity": NSNumber(value: table.capacity),
            "length": NSNumber(value: table.length),
            "tail": NSNumber(value: table.tail),
            "spread": NSNumber(value: table.spread),
        ]
    }

    /// Pull the pulse table of `id` and hand it to the bridge; `nil` = every pulse left the tips.
    private func uploadGeneratorFlow(id: String, into bridge: FilamentBridge) {
        guard generatorPulseLook[id] != nil else { return }
        let table = scene.generatorFlow(id: id).map(Self.pulseTableDictionary(from:))
        if !bridge.setGeneratedMeshFlow(id, table: table), table != nil {
            problem(.load, id, "the view rejected the pulse table (the trunk is not pulsed)")
        }
    }


    /// Enumerate `scene.listCameras()`, cache descriptors keyed by slug, and
    /// hand the bridge plain ObjC dictionaries. Each camera's pose lives on its
    /// baked carrier node; the bridge reuses the gltfio-born Camera component there.
    private func materializeCameras(into bridge: FilamentBridge) {
        let cameras = scene.listCameras()
        cameraDescriptors = Dictionary(
            cameras.map { ($0.id, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        bridge.addCameras(
            cameras.map {
                Self.cameraDictionary(
                    from: $0,
                    carrierGltfIndex: carrierIndexById[$0.id],
                    focusTargetGltfIndex: $0.focusTarget.flatMap { nodeIdToGltfIndex[$0] }
                )
            }
        )
    }

    /// Build the ObjC descriptor dictionary `FilamentBridge.addCameras(_:)`
    /// consumes: identity, projection kind + lens params, and the carrier index.
    static func cameraDictionary(
        from camera: TrideCameraDescriptor,
        carrierGltfIndex: Int? = nil,
        focusTargetGltfIndex: Int? = nil
    ) -> [String: Any] {
        var dict: [String: Any] = [
            "id": camera.id,
            "kind": camera.kind,
            "znear": camera.znear,
        ]
        if let yfov = camera.yfov { dict["yfov"] = yfov }
        if let aspect = camera.aspectRatio { dict["aspectRatio"] = aspect }
        if let xmag = camera.xmag { dict["xmag"] = xmag }
        if let ymag = camera.ymag { dict["ymag"] = ymag }
        if let zfar = camera.zfar { dict["zfar"] = zfar }
        if let idx = carrierGltfIndex { dict["carrierGltfIndex"] = idx }
        // The focus (a node wins over the distance) and the depth-of-field f-number.
        if let focus = camera.focusDistance { dict["focusDistance"] = focus }
        if let aperture = camera.aperture { dict["aperture"] = aperture }
        if let idx = focusTargetGltfIndex { dict["focusTargetGltfIndex"] = idx }
        return dict
    }

    /// Build the ObjC descriptor dictionary `FilamentBridge.addLights(_:)`
    /// consumes. Euler XYZ `rotation` is resolved to a unit `-Z` emission
    /// direction here so the bridge stays a pure Filament applier.
    static func lightDictionary(
        from light: TrideLightDescriptor,
        carrierGltfIndex: Int? = nil
    ) -> [String: Any] {
        var dict: [String: Any] = [
            "id": light.id,
            "type": light.lightType,
            "color": light.color,
            "intensity": light.intensity,
        ]
        if let range = light.range { dict["range"] = range }
        if let inner = light.innerConeAngle { dict["innerConeAngle"] = inner }
        if let outer = light.outerConeAngle { dict["outerConeAngle"] = outer }
        if let cast = light.castShadows { dict["castShadows"] = cast }
        if let position = light.position { dict["position"] = position }
        if let gltfIndex = light.gltfIndex { dict["gltfIndex"] = gltfIndex }
        // The baked carrier node — the bridge reuses its gltfio-born Light
        // component and leaves the pose to the TransformManager.
        if let idx = carrierGltfIndex { dict["carrierGltfIndex"] = idx }
        let direction = Self.lightForwardDirection(eulerXYZ: light.rotation)
        dict["direction"] = [direction.x, direction.y, direction.z]
        return dict
    }

    /// Resolve a light's Euler XYZ radians orientation to its world-space
    /// `-Z` emission direction (`KHR_lights_punctual`): compose the Euler
    /// to a quaternion (`Qx·Qy·Qz`) and rotate the forward axis.
    /// `nil` Euler = identity ⇒ forward `(0, 0, -1)`.
    static func lightForwardDirection(eulerXYZ euler: [Double]?) -> SIMD3<Float> {
        let forward = SIMD3<Float>(0, 0, -1)
        guard let euler, euler.count >= 3 else { return forward }
        let q = eulerXYZQuaternion(SIMD3(Float(euler[0]), Float(euler[1]), Float(euler[2])))
        return q.act(forward)
    }

    /// Apply a tick's mutations to Filament. Nodes dirty in 2+ TRS
    /// components use a single coalesced `setNodeTRS` bridge call;
    /// single-dirty nodes keep per-component calls.
    ///
    /// Transform dispatch uses **index-based** bridge calls
    /// (`setNodeRotation(atIndex:…)`, `setNodeScale(atIndex:…)`,
    /// `setNodeTranslation(atIndex:…)`), resolved through the bridge's node-index
    /// map (glTF `nodes[]` order, built from skin joints + name matching). This
    /// correctly handles duplicate glTF node names (e.g. nine petals sharing one
    /// name at different glTF indices).
    ///
    /// `complete`: the batch is the core's whole frame (the first after a reset) — an animated
    /// render setting it does not carry returns to the scene's own value.
    func dispatch(mutations: [TrideMutation], bridge: FilamentBridge, complete: Bool = false) {

        var dirtyRotations: Set<Int> = []
        var dirtyScales: Set<Int> = []
        var dirtyTranslations: Set<Int> = []
        var dirtyMaterials: Set<Int> = []
        var morphWeights: [Int: [Int: Float]] = [:]

        // Typed values from the mutations avoid a readProperty FFI
        // round-trip. Key = node index. Last-write-wins for duplicates within tick.
        var rotationValues: [Int: simd_quatf] = [:]
        var scaleValues: [Int: SIMD3<Float>] = [:]
        var translationValues: [Int: SIMD3<Float>] = [:]

        // Manifest-owned light channel mutations, keyed by slug.
        var lightAccumulators: [String: LightAccumulator] = [:]

        // Animatable HDR env channels (single environment per scene).
        var environmentAcc: EnvironmentAccumulator?
        var renderSettingsAcc: RenderSettingsAccumulator?

        // Carriers whose held pose this batch moved, and which of their components.
        var movedCarriers: [String: FilamentBridgeTRSMask] = [:]
        if complete {
            // A whole frame: a pose the core no longer writes is the authored one again.
            for (id, authored) in authoredCarrierPoses {
                guard let held = carrierPoses[id], held != authored else { continue }
                var mask: FilamentBridgeTRSMask = []
                if held.position != authored.position { mask.insert(.translation) }
                if held.euler != authored.euler { mask.insert(.rotation) }
                movedCarriers[id] = mask
                carrierPoses[id] = authored
            }
        }
        var cameraLenses: [String: CameraLensAccumulator] = [:]
        // `.some(nil)` = a switch to the default camera was requested.
        var activeCameraSwitch: String?? = nil
        // Generators whose geometry was announced this tick (ordered, deduped).
        var dirtyGenerators: [String] = []
        var dirtyDatablocks: [String] = []
        // Nodes announced bent anew (or unbent) this tick, ordered, deduped.
        var dirtyDeformed: [String] = []
        // `trunkEpoch` per announced generator (absent from an older core → upload).
        var announcedTrunk: [String: UInt64] = [:]
        // Palette looks a timeline / binding wrote this tick (ordered, deduped), and those written
        // in a leaf other than a node-graph input.
        var dirtyPalettes: [String] = []
        var writtenLooks: Set<String> = []
        // Generators whose pulse table was announced this tick.
        var dirtyFlows: [String] = []
        // A node's own visibility was written this tick.
        var visibilityDirty = false


        for mutation in mutations {
            switch kind(of: mutation.pointer) {
            case .nodeRotation(let i):
                dirtyRotations.insert(i)
                if let q = Self.extractQuaternion(from: mutation.typedValue) {
                    rotationValues[i] = q
                    // The last write wins: the cached Euler lanes no longer describe the node.
                    nodeEulerByIndex[i] = nil
                }
            case .nodeScale(let i):
                dirtyScales.insert(i)
                if let v = Self.extractVec3(from: mutation.typedValue) {
                    scaleValues[i] = v
                }
            case .nodeTranslation(let i):
                dirtyTranslations.insert(i)
                if let v = Self.extractVec3(from: mutation.typedValue) {
                    translationValues[i] = v
                }
            case .nodeTranslationComponent(let i, let c):
                dirtyTranslations.insert(i)
                if let s = Self.extractScalar(from: mutation.typedValue) {
                    var current = translationValues[i] ?? readVec3(pointer: "/nodes/\(i)/translation") ?? .zero
                    current[c] = Float(s)
                    translationValues[i] = current
                }
            case .nodeScaleComponent(let i, let c):
                dirtyScales.insert(i)
                if let s = Self.extractScalar(from: mutation.typedValue) {
                    var current = scaleValues[i] ?? readVec3(pointer: "/nodes/\(i)/scale") ?? SIMD3<Float>(1, 1, 1)
                    current[c] = Float(s)
                    scaleValues[i] = current
                }
            case .nodeRotationComponent(let i, let c):
                dirtyRotations.insert(i)
                if let s = Self.extractScalar(from: mutation.typedValue) {
                    if c < 3 {
                        var euler = nodeEulerByIndex[i]
                            ?? readRotationQuaternion(forNodeIndex: i).map(Self.eulerXYZ(from:))
                            ?? .zero
                        euler[c] = Float(s)
                        nodeEulerByIndex[i] = euler
                        rotationValues[i] = Self.eulerXYZQuaternion(euler)
                    } else {
                        // `rotation.w` has no Euler meaning: the raw quaternion component.
                        let base = rotationValues[i] ?? readRotationQuaternion(forNodeIndex: i) ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
                        var components = simd_float4(base.imag.x, base.imag.y, base.imag.z, base.real)
                        components[c] = Float(s)
                        rotationValues[i] = simd_quatf(ix: components[0], iy: components[1], iz: components[2], r: components[3])
                        nodeEulerByIndex[i] = nil
                    }
                }
            case .nodeMorphWeight(let ni, let wi):
                if let s = Self.extractScalar(from: mutation.typedValue) {
                    morphWeights[ni, default: [:]][wi] = Float(s)
                } else {
                    let v = readScalar(pointer: mutation.pointer) ?? 0
                    morphWeights[ni, default: [:]][wi] = v
                }
            case .nodeVisible(let i):
                if case .boolValue(let visible) = mutation.typedValue {
                    nodeVisibleOwn[i] = visible
                    visibilityDirty = true
                }
            case .materialBaseColor(let i):  dirtyMaterials.insert(i)
            case .light(let id, let channel):
                if let spatial = Self.spatialChannel(of: channel),
                   let expected = Self.wholeSpatialShapeProblem(spatial, mutation.typedValue) {
                    valueShapeProblem(mutation, expected)
                    continue
                }
                // A carrier-anchored light's pose lives on its carrier node;
                // only non-spatial channels overlay the light.
                if carrierIndexById[id] != nil, let spatial = Self.spatialChannel(of: channel) {
                    recordCarrierPose(id: id, channel: spatial, value: mutation.typedValue, moved: &movedCarriers)
                } else {
                    recordLightMutation(
                        id: id,
                        channel: channel,
                        value: mutation.typedValue,
                        into: &lightAccumulators
                    )
                }
            case .camera(let id, let channel):
                switch channel {
                case .position(let c), .rotation(let c):
                    let spatial: SpatialChannel
                    if case .position = channel {
                        spatial = .position(component: c)
                    } else {
                        spatial = .rotation(component: c)
                    }
                    if let expected = Self.wholeSpatialShapeProblem(spatial, mutation.typedValue) {
                        valueShapeProblem(mutation, expected)
                        continue
                    }
                    guard carrierIndexById[id] != nil else {
                        warnCarrierlessCameraOnce(id)
                        continue
                    }
                    recordCarrierPose(id: id, channel: spatial, value: mutation.typedValue, moved: &movedCarriers)
                case .yfov:
                    if let s = Self.extractScalar(from: mutation.typedValue) {
                        cameraLenses[id, default: CameraLensAccumulator()].yfov = Float(s)
                    }
                case .znear:
                    if let s = Self.extractScalar(from: mutation.typedValue) {
                        cameraLenses[id, default: CameraLensAccumulator()].znear = Float(s)
                    }
                case .zfar:
                    if let s = Self.extractScalar(from: mutation.typedValue) {
                        cameraLenses[id, default: CameraLensAccumulator()].zfar = Float(s)
                    }
                case .focusDistance:
                    if let s = Self.extractScalar(from: mutation.typedValue) {
                        cameraLenses[id, default: CameraLensAccumulator()].focusDistance = Float(s)
                    }
                case .aperture:
                    if let s = Self.extractScalar(from: mutation.typedValue) {
                        cameraLenses[id, default: CameraLensAccumulator()].aperture = Float(s)
                    }
                }
            case .activeCamera:
                activeCameraSwitch = .some(Self.extractString(from: mutation.typedValue))
            case .environment(let channel):
                recordEnvironmentMutation(
                    channel: channel,
                    value: mutation.typedValue,
                    into: &environmentAcc
                )
            case .renderSettings(let channel):
                var a = renderSettingsAcc ?? RenderSettingsAccumulator()
                switch channel {
                case .saturation:
                    if let v = Self.extractScalar(from: mutation.typedValue) { a.saturation = Float(v) }
                case .fogColor(.none):
                    if let c = Self.extractRgb(from: mutation.typedValue) { a.fogColor = c }
                case .fogColor(.some(let i)):
                    if let v = Self.extractScalar(from: mutation.typedValue) {
                        var base = a.fogColor ?? renderBaseFogColor
                        base[i] = Float(v)
                        a.fogColor = base
                    }
                case .bloomStrength:
                    if let v = Self.extractScalar(from: mutation.typedValue) { a.bloomStrength = Float(v) }
                case .pixelationEnabled:
                    if case .boolValue(let on) = mutation.typedValue {
                        a.pixelationEnabled = on
                    } else {
                        valueShapeProblem(mutation, "pixelation on / off expects true or false")
                    }
                case .pixelationSize:
                    if let v = Self.extractScalar(from: mutation.typedValue), v.isFinite {
                        a.pixelationSize = Float(v)
                    } else {
                        valueShapeProblem(mutation, "pixelation size expects a number")
                    }
                }
                renderSettingsAcc = a
            case .generatorGeometry(let id):
                if !dirtyGenerators.contains(id) { dirtyGenerators.append(id) }
                if let trunk = Self.trunkEpoch(of: mutation.typedValue) { announcedTrunk[id] = trunk }
            case .generatorFlow(let id):
                if !dirtyFlows.contains(id) { dirtyFlows.append(id) }
            case .datablockGeometry(let uid):
                if !dirtyDatablocks.contains(uid) { dirtyDatablocks.append(uid) }
            case .deformedGeometry(let uid):
                if !dirtyDeformed.contains(uid) { dirtyDeformed.append(uid) }
            case .materialOverride(let slug):
                if !dirtyPalettes.contains(slug) { dirtyPalettes.append(slug) }
                if complete || !Self.isGraphLeaf(mutation.pointer) { writtenLooks.insert(slug) }
                if let field = Self.glassField(of: mutation.pointer) { warnGlassOnce(slug: slug, field: field) }
            case .unknown:
                // The core also writes pointers it consumes itself (generator / effector parameters —
                // it re-evaluates and announces geometry); only a pointer in a domain the view draws,
                // that no path here takes, is a problem worth the host's attention.
                if Self.isViewDomain(mutation.pointer), unknownPointers.insert(mutation.pointer).inserted {
                    problem(.pointer, mutation.pointer, "the view has no path for this pointer — ignored")
                }
            }
        }


        // Nodes dirty in 2+ TRS components → single coalesced
        // setNodeTRS bridge call (1× lookup + compose + setTransform
        // instead of N×). Uses index-based dispatch to correctly
        // handle duplicate node names.
        let multiDirty = dirtyRotations.intersection(dirtyScales)
            .union(dirtyRotations.intersection(dirtyTranslations))
            .union(dirtyScales.intersection(dirtyTranslations))

        for i in multiDirty {
            var mask: FilamentBridgeTRSMask = []
            var t = SIMD3<Float>.zero
            var r = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            var s = SIMD3<Float>(1, 1, 1)

            if dirtyTranslations.contains(i) {
                if let tv = translationValues[i] ?? readVec3(pointer: "/nodes/\(i)/translation") {
                    t = tv
                    mask.insert(.translation)
                }
            }
            if dirtyRotations.contains(i) {
                if let q = rotationValues[i] ?? readRotationQuaternion(forNodeIndex: i) {
                    r = q
                    mask.insert(.rotation)
                }
            }
            if dirtyScales.contains(i) {
                if let sv = scaleValues[i] ?? readVec3(pointer: "/nodes/\(i)/scale") {
                    s = sv
                    mask.insert(.scale)
                }
            }

            guard !mask.isEmpty else { continue }

            bridge.setNodeTRS(atIndex: Int(i),
                              translation: t,
                              rotation: simd_float4(r.imag.x, r.imag.y, r.imag.z, r.real),
                              scale: s,
                              mask: mask)
        }

        // Single-component dispatches use index-based bridge calls —
        // correct for duplicate node names.
        // Prefer the mutation's typed value, fall back to the readProperty FFI.
        for i in dirtyRotations.subtracting(multiDirty) {
            guard let q = rotationValues[i] ?? readRotationQuaternion(forNodeIndex: i) else {
                logger.debug("rotation: node \(i) no value — skip")
                continue
            }
            bridge.setNodeRotation(atIndex: Int(i), quaternion: simd_float4(q.imag.x, q.imag.y, q.imag.z, q.real))
        }
        for i in dirtyScales.subtracting(multiDirty) {
            guard let s = scaleValues[i] ?? readVec3(pointer: "/nodes/\(i)/scale") else {
                logger.debug("scale: node \(i) no value — skip")
                continue
            }
            bridge.setNodeScale(atIndex: Int(i), x: s.x, y: s.y, z: s.z)
        }
        for i in dirtyTranslations.subtracting(multiDirty) {
            guard let t = translationValues[i] ?? readVec3(pointer: "/nodes/\(i)/translation") else {
                logger.debug("translation: node \(i) no value — skip")
                continue
            }
            bridge.setNodeTranslation(atIndex: Int(i), x: t.x, y: t.y, z: t.z)
        }
        for i in dirtyMaterials {
            guard let alpha = readMaterialAlpha(materialIndex: i) else { continue }
            for nodeIndex in materialIndexToNodeIndices[i] ?? [] {
                bridge.setNodeOpacity(atIndex: nodeIndex, opacity: alpha)
            }
        }

        // Morph target weights.
        for (nodeIndex, weights) in morphWeights {
            // Filament caps a renderable at 256 morph targets: a larger index is never a real lane
            // and must not size the held vector.
            let lanes = weights.filter { $0.key >= 0 && $0.key < 256 }
            let maxIdx = lanes.keys.max() ?? -1
            if maxIdx < 0 { continue }
            // Held, not rebuilt from the batch: lanes this tick did not write keep their value.
            // The bridge clamps to the renderable's target count.
            var arr = heldMorphWeights[nodeIndex] ?? []
            if arr.count < maxIdx + 1 {
                arr.append(contentsOf: [Float](repeating: 0, count: maxIdx + 1 - arr.count))
            }
            for (wi, val) in lanes {
                arr[wi] = val
            }
            heldMorphWeights[nodeIndex] = arr
            bridge.setMorphWeights(atIndex: nodeIndex, weights: &arr, count: arr.count)
        }

        // Apply accumulated light overlays. Routed by slug through
        // the bridge's single `slug → entity` lookup; rotation is resolved
        // Euler→quat→`-Z` here before reaching Filament.
        for (id, acc) in lightAccumulators {
            if let c = acc.color {
                bridge.setLightColor(id, r: c.x, g: c.y, b: c.z)
            }
            if let i = acc.intensity {
                bridge.setLightIntensity(id, intensity: i)
            }
            if let r = acc.range {
                bridge.setLightRange(id, range: r)
            }
            if acc.innerCone != nil || acc.outerCone != nil {
                let descriptor = lightDescriptors[id]
                let inner = acc.innerCone ?? liveCone[id]?.inner
                    ?? descriptor?.innerConeAngle.map(Float.init) ?? 0
                let outer = acc.outerCone ?? liveCone[id]?.outer
                    ?? descriptor?.outerConeAngle.map(Float.init) ?? Float.pi / 4
                bridge.setLightCone(id, inner: inner, outer: outer)
                liveCone[id] = (inner, outer)
            }
            if acc.rotationDirty, let e = acc.euler {
                let dir = Self.lightForwardDirection(
                    eulerXYZ: [Double(e.x), Double(e.y), Double(e.z)]
                )
                bridge.setLightDirection(id, direction: dir)
            }
            if let p = acc.position {
                bridge.setLightPosition(id, position: p)
            }
        }

        // Carrier-anchored light / camera poses land on the carrier node through
        // the SAME index-based TRS path mesh nodes use; the Light / Camera
        // components sit at identity under it, so the group transform composes
        // in Filament. The held pose goes whole for each component a lane moved; its rotation is
        // the core's composition of the held angles in the record's order.
        for (id, moved) in movedCarriers {
            guard !moved.isEmpty, let idx = carrierIndexById[id], let pose = carrierPoses[id] else { continue }
            var mask = moved
            var rotation = simd_float4(0, 0, 0, 1)
            if mask.contains(.rotation) {
                if let r = Self.carrierRotation(pose) {
                    rotation = r
                } else {
                    mask.remove(.rotation)
                    warnUnknownRotationOrderOnce(id, pose.order)
                    if mask.isEmpty { continue }
                }
            }
            bridge.setNodeTRS(atIndex: idx,
                              translation: pose.position,
                              rotation: rotation,
                              scale: SIMD3<Float>(1, 1, 1),
                              mask: mask)
        }
        for (id, lens) in cameraLenses {
            let d = cameraDescriptors[id]
            if lens.yfov != nil || lens.znear != nil || lens.zfar != nil {
                let live = liveProjection[id]
                let yfov = lens.yfov ?? live?.yfov ?? d?.yfov.map(Float.init) ?? 0
                let znear = lens.znear ?? live?.znear ?? Float(d?.znear ?? 0.1)
                let zfar = lens.zfar ?? live?.zfar ?? d?.zfar.map(Float.init) ?? 0
                bridge.setCameraProjection(id, yfov: yfov, znear: znear, zfar: zfar)
                liveProjection[id] = (yfov, znear, zfar)
            }
            if let focus = lens.focusDistance { bridge.setCameraFocusDistance(id, distance: focus) }
            if let aperture = lens.aperture { bridge.setCameraAperture(id, fNumber: aperture) }
        }
        if case .some(let slug) = activeCameraSwitch {
            bridge.setActiveCamera(slug)
        }

        // Apply accumulated HDR env overlay. Rotation is resolved Euler→matrix
        // at the bridge boundary (`eulerXYZRotationMatrix`). Both setters are
        // no-ops if no IBL is currently bound.
        if let acc = environmentAcc {
            if acc.rotationDirty, let e = acc.euler {
                bridge.setIndirectLightRotation(eulerXYZ: e)
            }
            if let i = acc.intensity {
                bridge.setIndirectLightIntensity(i)
            }
            // The background skybox recolours whether or not an IBL is bound.
            if let c = acc.backgroundColor {
                environmentBaseBackground = c
                bridge.setBackgroundColor(r: c.x, g: c.y, b: c.z)
            }
        }

        // A complete batch: what it does not carry is no longer animated — the scene's own value.
        if complete, let authored = authoredRender {
            var a = renderSettingsAcc ?? RenderSettingsAccumulator()
            if a.saturation == nil { a.saturation = authored.saturation }
            if a.fogColor == nil { a.fogColor = authored.fogColor }
            if a.bloomStrength == nil { a.bloomStrength = authored.bloomStrength }
            if a.pixelationEnabled == nil { a.pixelationEnabled = authored.pixelationEnabled }
            if a.pixelationSize == nil { a.pixelationSize = authored.pixelationSize }
            renderSettingsAcc = a
        }
        // Saturation / fog colour / bloom strength / pixelation once per tick. The bridge rebuilds
        // the LUT, re-sets the bloom and re-lays the blocks only when the value moved (a held clip
        // repeats it every tick).
        if let acc = renderSettingsAcc {
            if let s = acc.saturation {
                bridge.setSaturation(s)
            }
            if let c = acc.fogColor {
                renderBaseFogColor = c
                bridge.setFogColor(r: c.x, g: c.y, b: c.z)
            }
            if let b = acc.bloomStrength {
                bridge.setBloomStrength(b)
            }
            if let on = acc.pixelationEnabled {
                bridge.setPixelationEnabled(on)
            }
            if let size = acc.pixelationSize {
                bridge.setPixelationSize(size)
            }
        }

        for id in dirtyGenerators {
            if let trunk = announcedTrunk[id], uploadedTrunkEpoch[id] == trunk {
                syncModuleInstances(id: id, into: bridge)
            } else if uploadedTrunkEpoch[id] != nil {
                // A trunk already on screen takes its new mesh on the next tick: the core's
                // re-evaluation and the upload (~10 ms each for a large grown tree on an
                // iPhone 13) do not land in one frame. Its modules and pulses wait with it.
                deferredTrunks.removeAll { $0.id == id }
                deferredTrunks.append((id, announcedTrunk[id]))
            } else {
                uploadTrunk(id: id, trunk: announcedTrunk[id], into: bridge)
            }
        }
        // After the meshes: a trunk born this tick owns its flow material before its table lands.
        for id in dirtyFlows where !deferredTrunks.contains(where: { $0.id == id }) {
            uploadGeneratorFlow(id: id, into: bridge)
        }
        // A rebuilt parametric datablock — its nodes and its module mesh.
        for uid in dirtyDatablocks {
            refreshParametric(datablock: uid, into: bridge)
        }
        // After the rebuilds: a bent node draws what its deformers make of the new base.
        for uid in dirtyDeformed {
            refreshDeformed(node: uid, into: bridge)
        }
        // After the meshes too: a look recolours the instances this tick's uploads made.
        for slug in dirtyPalettes {
            recolourPalette(slug: slug, lookWritten: writtenLooks.contains(slug), into: bridge)
        }
        // Visibility last: a visibility write, or a mesh re-uploaded under a hidden host.
        if visibilityDirty
            || ((!dirtyGenerators.isEmpty || !dirtyDatablocks.isEmpty || !dirtyDeformed.isEmpty)
                && nodeVisibleOwn.values.contains(false))
        {
            applyNodeVisibility(to: bridge)
        }

    }

    // MARK: - Node visibility

    /// Set every node's effective visibility, parents first: the bridge applies a node's flag to
    /// its whole subtree, so a descendant set later keeps its own (hidden) state.
    private func applyNodeVisibility(to bridge: FilamentBridge) {
        let effective = Self.effectiveVisibility(own: nodeVisibleOwn, parent: nodeParentIndex)
        for index in Self.parentsFirst(nodeVisibleOwn.keys, parent: nodeParentIndex) {
            bridge.setNodeVisible(atIndex: index, visible: effective[index] ?? true)
        }
    }

    /// The node AND every ancestor (`nodeVisibleOwn` missing = visible) — the same
    /// effective visibility the editor draws.
    static func effectiveVisibility(own: [Int: Bool], parent: [Int: Int]) -> [Int: Bool] {
        var out: [Int: Bool] = [:]
        for index in own.keys {
            var visible = true
            var cursor: Int? = index
            var guardSteps = own.count + 1
            while let c = cursor, guardSteps > 0 {
                if own[c] == false { visible = false; break }
                cursor = parent[c]
                guardSteps -= 1
            }
            out[index] = visible
        }
        return out
    }

    /// Node indices ordered by depth (roots first), ties by index — deterministic.
    static func parentsFirst<S: Sequence>(_ nodes: S, parent: [Int: Int]) -> [Int] where S.Element == Int {
        func depth(_ index: Int) -> Int {
            var d = 0
            var cursor = parent[index]
            while let c = cursor, d <= parent.count {
                d += 1
                cursor = parent[c]
            }
            return d
        }
        return nodes.sorted { (depth($0), $0) < (depth($1), $1) }
    }

    // MARK: - Light mutation accumulation

    /// Fold one light-channel mutation into the per-slug accumulator.
    /// Whole-vector channels take the typed value verbatim; per-axis
    /// channels seed the other axes from the in-tick accumulator, then the
    /// cached descriptor, so the bridge always receives a complete vector.
    private func recordLightMutation(
        id: String,
        channel: LightChannel,
        value: TrideTypedValue,
        into accumulators: inout [String: LightAccumulator]
    ) {
        var acc = accumulators[id] ?? LightAccumulator()
        let descriptor = lightDescriptors[id]

        switch channel {
        case .color(.none):
            if let v = Self.extractVec3(from: value) { acc.color = v }
        case .color(.some(let c)):
            if let s = Self.extractScalar(from: value) {
                var base = acc.color ?? descriptorVec3(descriptor?.color) ?? SIMD3(1, 1, 1)
                base[c] = Float(s)
                acc.color = base
            }
        case .intensity:
            if let s = Self.extractScalar(from: value) { acc.intensity = Float(s) }
        case .range:
            if let s = Self.extractScalar(from: value) { acc.range = Float(s) }
        case .innerCone:
            if let s = Self.extractScalar(from: value) { acc.innerCone = Float(s) }
        case .outerCone:
            if let s = Self.extractScalar(from: value) { acc.outerCone = Float(s) }
        case .rotation(.none):
            if let v = Self.extractVec3(from: value) {
                acc.euler = v
                acc.rotationDirty = true
            }
        case .rotation(.some(let c)):
            if let s = Self.extractScalar(from: value) {
                var base = acc.euler ?? descriptorVec3(descriptor?.rotation) ?? .zero
                base[c] = Float(s)
                acc.euler = base
                acc.rotationDirty = true
            }
        case .position(.none):
            if let v = Self.extractVec3(from: value) { acc.position = v }
        case .position(.some(let c)):
            if let s = Self.extractScalar(from: value) {
                var base = acc.position ?? descriptorVec3(descriptor?.position) ?? .zero
                base[c] = Float(s)
                acc.position = base
            }
        }

        accumulators[id] = acc
    }

    private func descriptorVec3(_ values: [Double]?) -> SIMD3<Float>? {
        guard let values, values.count >= 3 else { return nil }
        return SIMD3(Float(values[0]), Float(values[1]), Float(values[2]))
    }

    /// The spatial lane of a light channel (position / rotation), or `nil` for
    /// the non-spatial ones that always overlay the LightManager component.
    static func spatialChannel(of channel: LightChannel) -> SpatialChannel? {
        switch channel {
        case .position(let c): return .position(component: c)
        case .rotation(let c): return .rotation(component: c)
        default: return nil
        }
    }

    /// What a whole light / camera position or rotation expects when `value` is not it, else `nil`:
    /// the record's 3-vector (a rotation = its Euler angles, radians, the record's order — the core
    /// turns a quaternion into them). A quaternion is refused, never read as angles; a lane is not
    /// judged here.
    static func wholeSpatialShapeProblem(_ channel: SpatialChannel, _ value: TrideTypedValue) -> String? {
        switch channel {
        case .position(.none):
            return extractVec3(from: value) == nil ? "a whole position expects [x, y, z]" : nil
        case .rotation(.none):
            return extractVec3(from: value) == nil
                ? "a whole rotation expects the stored Euler angles [x, y, z] (radians); a quaternion is converted by the core"
                : nil
        case .position(.some), .rotation(.some):
            return nil
        }
    }

    /// Fold one spatial mutation of a carrier-anchored entity into its held LOCAL
    /// pose: a whole vector replaces the component, a per-axis lane patches that axis and the
    /// others keep what the earlier batches left — never the authored value (changed-only
    /// batches leave out an axis the core stopped writing). Euler-space patching, no quaternion
    /// re-decompose (multi-turn faithful).
    private func recordCarrierPose(
        id: String,
        channel: SpatialChannel,
        value: TrideTypedValue,
        moved: inout [String: FilamentBridgeTRSMask]
    ) {
        var pose = carrierPoses[id] ?? CarrierPose(position: .zero, euler: .zero)
        switch channel {
        case .position(.none):
            guard let v = Self.extractVec3(from: value) else { return }
            pose.position = v
            moved[id, default: []].insert(.translation)
        case .position(.some(let c)):
            guard let s = Self.extractScalar(from: value) else { return }
            pose.position[c] = Float(s)
            moved[id, default: []].insert(.translation)
        case .rotation(.none):
            guard let v = Self.extractVec3(from: value) else { return }
            pose.euler = v
            moved[id, default: []].insert(.rotation)
        case .rotation(.some(let c)):
            guard let s = Self.extractScalar(from: value) else { return }
            pose.euler[c] = Float(s)
            moved[id, default: []].insert(.rotation)
        }
        carrierPoses[id] = pose
    }

    /// Warn-once notice: a camera pose mutation with no baked carrier
    /// (archive predates `carrierMetadata`) cannot be posed on this build.
    private func warnCarrierlessCameraOnce(_ id: String) {
        guard !warnedCarrierlessCameras.contains(id) else { return }
        warnedCarrierlessCameras.insert(id)
        problem(.pointer, id, "camera has no carrier node — its movement is dropped; re-export the scene")
    }

    /// The quaternion `[x, y, z, w]` of a carrier's held angles in its record's order — the
    /// core's one composition (`eulerToQuat`, the same the core's own camera pose and `pick`
    /// use), so any order the core knows places the carrier alike on every host. `nil` = an
    /// order this core does not know (a host built against another core).
    static func carrierRotation(_ pose: CarrierPose) -> simd_float4? {
        let e = pose.euler
        let q = eulerToQuat(euler: [Double(e.x), Double(e.y), Double(e.z)], order: pose.order)
        guard q.count == 4 else { return nil }
        return simd_float4(Float(q[0]), Float(q[1]), Float(q[2]), Float(q[3]))
    }

    /// Warn-once notice: the core refused a carrier's rotation order — the carrier keeps its
    /// last rotation (no guessed composition).
    private func warnUnknownRotationOrderOnce(_ id: String, _ order: String) {
        guard !warnedUnknownRotationOrders.contains(id) else { return }
        warnedUnknownRotationOrders.insert(id)
        problem(.pointer, id, "rotation order '\(order)' is unknown to the core — the rotation is not applied")
    }

    // MARK: - Environment mutation accumulation

    /// Fold one HDR-env channel mutation into the per-tick accumulator. The
    /// whole-vec3 rotation channel takes the typed value verbatim; a per-axis
    /// channel (`rotation.y`) seeds the other axes from the in-tick accumulator,
    /// then `environmentBaseEuler`, so the bridge always receives a full Euler.
    private func recordEnvironmentMutation(
        channel: EnvironmentChannel,
        value: TrideTypedValue,
        into acc: inout EnvironmentAccumulator?
    ) {
        var a = acc ?? EnvironmentAccumulator()
        switch channel {
        case .intensity:
            if let s = Self.extractScalar(from: value) { a.intensity = Float(s) }
        case .rotation(.none):
            if let v = Self.extractVec3(from: value) {
                a.euler = v
                a.rotationDirty = true
            }
        case .rotation(.some(let c)):
            if let s = Self.extractScalar(from: value) {
                var base = a.euler ?? environmentBaseEuler
                base[c] = Float(s)
                a.euler = base
                a.rotationDirty = true
            }
        case .backgroundColor(.none):
            if let v = Self.extractRgb(from: value) { a.backgroundColor = v }
        case .backgroundColor(.some(let c)):
            if let s = Self.extractScalar(from: value) {
                var base = a.backgroundColor ?? environmentBaseBackground
                base[c] = Float(s)
                a.backgroundColor = base
            }
        }
        acc = a
    }

    // MARK: - Pointer parsing

    /// Domains whose pointers the view applies itself (as opposed to the core's own parameters).
    private static let viewDomains: Set<Substring> = [
        "node", "nodes", "materials", "materialOverrides", "lights", "cameras", "environment_settings",
        "render_settings", "scene",
    ]

    static func isViewDomain(_ pointer: String) -> Bool {
        pointer.split(separator: "/", maxSplits: 1).first.map(viewDomains.contains) ?? false
    }

    /// Filament's morph-target limit per primitive: a weight index beyond it addresses nothing.
    static let maxMorphTargets = 256

    /// Classify a JSON Pointer string into one of the dispatch buckets.
    /// Static + pure to keep the unit tests trivial.
    static func classify(_ pointer: String) -> DispatchKind {
        guard let segs = parsePointer(pointer), segs.count >= 2 else {
            return .unknown
        }
        // `/environment_settings/<channel>` is a flat 2-segment
        // manifest namespace (not under `/scene/`), so it is matched before the
        // 3-segment node/material/light forms below.
        // `/render_settings/tone_mapping/saturation`, `/render_settings/bloom/strength`,
        // `/render_settings/pixelation/{enabled,size}`, `/render_settings/fog/color` and a fog colour
        // lane `/render_settings/fog/color/{0..2|r|g|b}`.
        if segs[0] == "render_settings" {
            if segs.count == 3, segs[1] == "tone_mapping", segs[2] == "saturation" {
                return .renderSettings(channel: .saturation)
            }
            if segs.count == 3, segs[1] == "bloom", segs[2] == "strength" {
                return .renderSettings(channel: .bloomStrength)
            }
            if segs.count == 3, segs[1] == "pixelation" {
                switch segs[2] {
                case "enabled": return .renderSettings(channel: .pixelationEnabled)
                case "size": return .renderSettings(channel: .pixelationSize)
                default: return .unknown
                }
            }
            guard segs.count >= 3, segs.count <= 4, segs[1] == "fog" else { return .unknown }
            // The editor's grammar too: a lane is `/color/1` or the dotted `/color.g`.
            let leaf = segs[2].split(separator: ".", maxSplits: 1).map(String.init)
            guard leaf.first == "color" else { return .unknown }
            let lane = segs.count == 4 ? segs[3] : (leaf.count == 2 ? leaf[1] : nil)
            if segs.count == 4, leaf.count == 2 { return .unknown }
            guard let lane else { return .renderSettings(channel: .fogColor(component: nil)) }
            guard let c = rgbComponentIndex(lane) else { return .unknown }
            return .renderSettings(channel: .fogColor(component: c))
        }
        if segs[0] == "environment_settings" {
            // A slash lane (`rotation/y`, `background_color/1` — the catalogue's lanes) is the
            // dotted one spelled with a slash, folded as for lights and cameras.
            let lane: String?
            switch segs.count {
            case 2: lane = segs[1]
            case 3: lane = dottedLane(segs[1], axis: segs[2])
            default: lane = nil
            }
            guard let lane, let channel = classifyEnvironmentChannel(lane) else {
                return .unknown
            }
            return .environment(channel: channel)
        }
        // `/scene/active_camera` — 2-segment scene-domain
        // switch (hard cut). Other `/scene/*` pointers are not bridge-routable.
        if segs[0] == "scene" {
            return segs.count == 2 && segs[1] == "active_camera" ? .activeCamera : .unknown
        }
        guard segs.count >= 3 else { return .unknown }
        switch segs[0] {
        case "nodes":
            // Indices come from a `.trd` the host did not write: out of range = unknown, never a
            // subscript that traps in the frame.
            guard let idx = Int(segs[1]), idx >= 0 else { return .unknown }
            switch segs[2] {
            case "rotation":
                // A slash-LETTER axis (`/rotation/x`) is the same Euler alias as the dot form —
                // the core's bindings write path emits it. Without it a 4-segment pointer would
                // update the WHOLE rotation from a lone scalar (a silent no-op: `extractQuaternion`
                // rejects a bare number). A numeric slash (`/rotation/0`) addresses a raw
                // quaternion component and stays on the whole-vector path below.
                if segs.count == 4, let c = xyzwComponentIndex(segs[3]) {
                    return .nodeRotationComponent(index: idx, component: c)
                }
                return .nodeRotation(index: idx)
            case "scale":
                // A numeric axis (`/scale/1`) writes one component; the whole-vector path would
                // broadcast the lone scalar to all three axes.
                if segs.count == 4 {
                    guard let c = Int(segs[3]), (0..<3).contains(c) else { return .unknown }
                    return .nodeScaleComponent(index: idx, component: c)
                }
                return .nodeScale(index: idx)
            case "translation":
                if segs.count == 4 {
                    guard let c = Int(segs[3]), (0..<3).contains(c) else { return .unknown }
                    return .nodeTranslationComponent(index: idx, component: c)
                }
                return .nodeTranslation(index: idx)
            case "weights":
                guard segs.count >= 4, let wi = Int(segs[3]), (0..<Self.maxMorphTargets).contains(wi) else {
                    return .unknown
                }
                return .nodeMorphWeight(nodeIndex: idx, weightIndex: wi)
            case "visible":
                return segs.count == 3 ? .nodeVisible(index: idx) : .unknown
            default:
                if let kind = classifyDotNotation(segs[2], nodeIndex: idx) {
                    return kind
                }
                return .unknown
            }
        case "materialOverrides":
            guard segs.count >= 3, !segs[1].isEmpty else { return .unknown }
            return .materialOverride(slug: segs[1])
        case "materials":
            guard let idx = Int(segs[1]), idx >= 0 else { return .unknown }
            return segs.contains("baseColorFactor")
                ? .materialBaseColor(index: idx)
                : .unknown
        case "generators":
            guard segs.count == 3 else { return .unknown }
            switch segs[2] {
            case "geometry": return .generatorGeometry(id: segs[1])
            case "flow": return .generatorFlow(id: segs[1])
            default: return .unknown
            }
        case "datablocks":
            guard segs.count == 3, segs[2] == "geometry", !segs[1].isEmpty else { return .unknown }
            return .datablockGeometry(uid: segs[1])
        case "deformed":
            guard segs.count == 3, segs[2] == "geometry", !segs[1].isEmpty else { return .unknown }
            return .deformedGeometry(nodeUid: segs[1])
        case "lights":
            // segs[1] is a manifest slug (string), not an integer index.
            let id = segs[1]
            guard !id.isEmpty, let lane = entityLane(segs),
                  let channel = classifyLightChannel(lane) else {
                return .unknown
            }
            return .light(id: id, channel: channel)
        case "cameras":
            // Manifest camera slug + property channel.
            let id = segs[1]
            guard !id.isEmpty, let lane = entityLane(segs),
                  let channel = classifyCameraChannel(lane) else {
                return .unknown
            }
            return .camera(id: id, channel: channel)
        default:
            return .unknown
        }
    }

    /// The property lane of a `/cameras/{slug}/…` or `/lights/{slug}/…` pointer, a slash axis folded
    /// into the dot form the channel parsers read (`dottedLane`). A bindings-graph wire writes the
    /// slash form, a timeline the dot form; both are read, as in the editor. Without the fold a lone
    /// axis value would hit the whole-vector channel and be dropped.
    private static func entityLane(_ segs: [String]) -> String? {
        switch segs.count {
        case 3: return segs[2]
        case 4: return dottedLane(segs[2], axis: segs[3])
        default: return nil
        }
    }

    /// `property` + a slash axis → the dot form: `position` + `1` or `y` → `position.y`,
    /// `color` + `1` → `color.y` (= green), `color` + `g` → `color.g`; any other axis → `nil`.
    private static func dottedLane(_ property: String, axis: String) -> String? {
        switch axis {
        case "0", "x": return "\(property).x"
        case "1", "y": return "\(property).y"
        case "2", "z": return "\(property).z"
        case "r", "g", "b": return "\(property).\(axis)"
        default: return nil
        }
    }

    /// Map a `/cameras/{slug}/…` property segment to a `CameraChannel`.
    /// Whole-vector (`position` / `rotation`), dotted per-axis (`rotation.y`),
    /// the editor grammar's `translation` alias, and the scalar
    /// lens lanes (`yfov` / `znear` / `zfar` / `focusDistance` / `aperture`).
    private static func classifyCameraChannel(_ segment: String) -> CameraChannel? {
        if let dotIdx = segment.firstIndex(of: ".") {
            let base = String(segment[segment.startIndex..<dotIdx])
            let comp = String(segment[segment.index(after: dotIdx)...])
            guard let c = xyzComponentIndex(comp) else { return nil }
            switch base {
            case "position", "translation": return .position(component: c)
            case "rotation":                return .rotation(component: c)
            default:                        return nil
            }
        }
        switch segment {
        case "position", "translation": return .position(component: nil)
        case "rotation":                return .rotation(component: nil)
        case "yfov":                    return .yfov
        case "znear":                   return .znear
        case "zfar":                    return .zfar
        case "focusDistance":           return .focusDistance
        case "aperture":                return .aperture
        default:                        return nil
        }
    }

    /// Map a `/lights/{slug}/…` property segment to a `LightChannel`.
    /// Handles both whole-vector (`rotation`) and dotted per-axis
    /// (`rotation.x`, `color.g`) forms emitted by the runtime; `translation` (the catalogue's
    /// name, as for cameras) is the manifest's `position`.
    private static func classifyLightChannel(_ segment: String) -> LightChannel? {
        if let dotIdx = segment.firstIndex(of: ".") {
            let base = String(segment[segment.startIndex..<dotIdx])
            let comp = String(segment[segment.index(after: dotIdx)...])
            switch base {
            case "color":
                guard let c = rgbComponentIndex(comp) else { return nil }
                return .color(component: c)
            case "rotation":
                guard let c = xyzComponentIndex(comp) else { return nil }
                return .rotation(component: c)
            case "position", "translation":
                guard let c = xyzComponentIndex(comp) else { return nil }
                return .position(component: c)
            default:
                return nil
            }
        }
        switch segment {
        case "color":          return .color(component: nil)
        case "intensity":      return .intensity
        case "range":          return .range
        case "innerConeAngle": return .innerCone
        case "outerConeAngle": return .outerCone
        case "rotation":       return .rotation(component: nil)
        case "position", "translation": return .position(component: nil)
        default:               return nil
        }
    }

    /// Map an `/environment_settings/<segment>` property to an
    /// `EnvironmentChannel`. Handles whole-vec3 (`rotation`), per-axis dotted
    /// (`rotation.y`), scalar (`intensity`) and the whole / one-channel background
    /// colour (`background_color`, `background_color.g`) forms the runtime
    /// emits. `background_visible` (and any other segment) returns `nil` →
    /// `.unknown`, since it is not animatable.
    private static func classifyEnvironmentChannel(_ segment: String) -> EnvironmentChannel? {
        if let dotIdx = segment.firstIndex(of: ".") {
            let base = String(segment[segment.startIndex..<dotIdx])
            let comp = String(segment[segment.index(after: dotIdx)...])
            switch base {
            case "rotation":
                guard let c = xyzComponentIndex(comp) else { return nil }
                return .rotation(component: c)
            case "background_color":
                guard let c = rgbComponentIndex(comp) else { return nil }
                return .backgroundColor(component: c)
            default:
                return nil
            }
        }
        switch segment {
        case "rotation":         return .rotation(component: nil)
        case "intensity":        return .intensity
        case "background_color": return .backgroundColor(component: nil)
        default:                 return nil
        }
    }

    private static func rgbComponentIndex(_ s: String) -> Int? {
        switch s {
        case "r", "x", "0": return 0
        case "g", "y", "1": return 1
        case "b", "z", "2": return 2
        default: return nil
        }
    }

    private static func classifyDotNotation(_ segment: String, nodeIndex idx: Int) -> DispatchKind? {
        guard let dotIdx = segment.firstIndex(of: ".") else { return nil }
        let base = String(segment[segment.startIndex..<dotIdx])
        let comp = String(segment[segment.index(after: dotIdx)...])

        switch base {
        case "translation":
            guard let c = xyzComponentIndex(comp) else { return nil }
            return .nodeTranslationComponent(index: idx, component: c)
        case "scale":
            guard let c = xyzComponentIndex(comp) else { return nil }
            return .nodeScaleComponent(index: idx, component: c)
        case "rotation":
            guard let c = xyzwComponentIndex(comp) else { return nil }
            return .nodeRotationComponent(index: idx, component: c)
        default:
            return nil
        }
    }

    private static func xyzComponentIndex(_ s: String) -> Int? {
        switch s {
        case "x": return 0
        case "y": return 1
        case "z": return 2
        default: return nil
        }
    }

    private static func xyzwComponentIndex(_ s: String) -> Int? {
        switch s {
        case "x": return 0
        case "y": return 1
        case "z": return 2
        case "w": return 3
        default: return nil
        }
    }


    /// RFC 6901 pointer parser. Returns `nil` for malformed pointers
    /// (missing leading `/`); returns `[]` for the root pointer `""`.
    static func parsePointer(_ pointer: String) -> [String]? {
        if pointer.isEmpty { return [] }
        guard pointer.hasPrefix("/") else { return nil }
        let raw = pointer.dropFirst().split(separator: "/", omittingEmptySubsequences: false)
        return raw.map { decodeSegment(String($0)) }
    }

    /// RFC 6901 §4 escapes: decode `~1` to `/` first, then `~0` to `~`.
    static func decodeSegment(_ raw: String) -> String {
        var s = raw.replacingOccurrences(of: "~1", with: "/")
        s = s.replacingOccurrences(of: "~0", with: "~")
        return s
    }

    // MARK: - Node pointer resolution

    /// Rewrite a canonical `/node/{uid}/...` mutation pointer to the
    /// index-based `/nodes/{gltfIndex}/...` form the dispatch classifier and
    /// the Filament bridge consume. The
    /// FFI emits **every** node mutation — runtime overrides and
    /// authored-timeline alike — as `/node/{uid}`, so all incoming pointers
    /// flow through here.
    private func resolveNodePointer(_ pointer: String) -> String {
        Self.rewriteNodePointer(pointer, using: nodeIdToGltfIndex)
    }

    /// `classify` of the resolved pointer, parsed once per pointer (`kindByPointer`).
    private func kind(of pointer: String) -> DispatchKind {
        if let known = kindByPointer[pointer] { return known }
        let kind = Self.classify(resolveNodePointer(pointer))
        kindByPointer[pointer] = kind
        return kind
    }

    /// Pure `/node/{uid}` → `/nodes/{gltfIndex}` rewrite, factored static so
    /// `PointerDispatcherTests` can verify it without a live `TrideScene`.
    /// Non-`/node/` pointers (`/lights/…`, `/materials/…`, already-resolved
    /// `/nodes/…`) pass through unchanged; an unresolvable UID returns the
    /// original pointer (classifies as `.unknown` and is skipped).
    static func rewriteNodePointer(
        _ pointer: String,
        using nodeIdToGltfIndex: [String: Int]
    ) -> String {
        guard pointer.hasPrefix("/node/") else { return pointer }
        // Split: "/node/{uid}/property" → ["", "node", "{uid}/rest..."]
        let parts = pointer.split(separator: "/", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count >= 3 else { return pointer }
        let remainder = parts[2]
        // Split the UID from the rest: "{uid}/property" → uid + "/property".
        if let slashIdx = remainder.firstIndex(of: "/") {
            let nodeId = String(remainder[remainder.startIndex..<slashIdx])
            let rest = String(remainder[slashIdx...]) // includes leading "/"
            guard let gltfIndex = nodeIdToGltfIndex[nodeId] else {
                logger.debug("node '\(nodeId, privacy: .public)' not in nodeMetadata — skipping")
                return pointer
            }
            return "/nodes/\(gltfIndex)\(rest)"
        }
        // Pointer is just "/node/{uid}" with no property suffix.
        let nodeId = String(remainder)
        guard let gltfIndex = nodeIdToGltfIndex[nodeId] else {
            logger.debug("node '\(nodeId, privacy: .public)' not in nodeMetadata — skipping")
            return pointer
        }
        return "/nodes/\(gltfIndex)"
    }

    // MARK: - Value reading

    /// Read a node's rotation pointer and synthesise a `simd_quatf`.
    /// 4-array = canonical quaternion verbatim, 3-array = Euler XYZ radians
    /// from older manifests (warn-once).
    private func readRotationQuaternion(forNodeIndex idx: Int) -> simd_quatf? {
        let pointer = "/nodes/\(idx)/rotation"
        guard let array = readNumberArray(pointer: pointer) else { return nil }
        return Self.synthesiseQuaternion(from: array, nodeIndex: idx)
    }

    /// Synthesise a `simd_quatf` from a rotation pointer payload:
    ///
    /// - **4 components** → canonical quaternion `[qx, qy, qz, qw]`
    ///   read verbatim into `simd_quatf(ix:iy:iz:r:)`. This is the
    ///   atomic-write path emitted by the core's baked timeline and
    ///   by all core-driven runtime overrides (`setNodeRotation`).
    /// - **3 components** → Euler XYZ radians from older manifests
    ///   (manifest schema v3). Emits a warn-once notice on first
    ///   use per session.
    /// - **Any other length** → error log, returns `nil` (caller skips
    ///   the rotation update for this tick).
    static func synthesiseQuaternion(from array: [Float], nodeIndex: Int? = nil) -> simd_quatf? {
        switch array.count {
        case 4:
            return simd_quatf(ix: array[0], iy: array[1], iz: array[2], r: array[3])
        case 3:
            warnOnceLegacyEulerRotation(nodeIndex: nodeIndex)
            return eulerXYZQuaternion(SIMD3(array[0], array[1], array[2]))
        default:
            logger.error("rotation array length \(array.count) unsupported")
            return nil
        }
    }

    /// Notice for a 3-component Euler rotation value. Fires once per session
    /// on the first such read; subsequent reads stay quiet so noisy v3
    /// manifests don't flood OSLog.
    private static func warnOnceLegacyEulerRotation(nodeIndex: Int?) {
        guard !legacyEulerRotationWarned else { return }
        legacyEulerRotationWarned = true
        let nodeRef = nodeIndex.map { "/nodes/\($0)" } ?? "<unknown node>"
        logger.warning(
            """
            3-component Euler XYZ rotation read at \
            \(nodeRef, privacy: .public)/rotation; this fallback is kept \
            for older manifests only. Re-export the \
            scene under manifest v4+ to emit canonical 4-component \
            quaternions and silence this notice.
            """
        )
    }

    /// The quaternion of Euler XYZ angles in radians (`Q = Qx · Qy · Qz`, the editor's Euler XYZ)
    /// — the core's composition (`eulerToQuat`), never one of the view's own. Used by per-axis node
    /// rotation channels, the directions of carrier-less lights and 3-component rotation values;
    /// canonical 4-component quaternions bypass it, a carrier composes in its record's order
    /// (`carrierRotation`).
    static func eulerXYZQuaternion(_ e: SIMD3<Float>) -> simd_quatf {
        let q = eulerToQuat(euler: [Double(e.x), Double(e.y), Double(e.z)], order: eulerXYZOrder)
        // INVARIANT: "xyz" and three angles are always the core's; an empty answer is a core this
        // view was not built against — said, and the identity drawn.
        guard q.count == 4 else {
            logger.error("eulerToQuat refused the xyz order — drawn unrotated")
            return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        }
        return simd_quatf(ix: Float(q[0]), iy: Float(q[1]), iz: Float(q[2]), r: Float(q[3]))
    }

    /// Euler XYZ angles in radians of `q` — the core's decomposition (`quatToEuler`: on a gimbal
    /// pole, Y = ±90°, the coupled X ± Z goes to X), so the per-axis Euler state a node's lanes are
    /// seeded with from its rest quaternion composes back to that pose on every host.
    static func eulerXYZ(from q: simd_quatf) -> SIMD3<Float> {
        let e = quatToEuler(
            q: [Double(q.imag.x), Double(q.imag.y), Double(q.imag.z), Double(q.real)], order: eulerXYZOrder)
        // INVARIANT: as in `eulerXYZQuaternion`.
        guard e.count == 3 else {
            logger.error("quatToEuler refused the xyz order — lanes seeded at zero")
            return .zero
        }
        return SIMD3(Float(e[0]), Float(e[1]), Float(e[2]))
    }

    // MARK: - typedValue extraction

    /// Extract a quaternion from TrideTypedValue. Returns nil if value
    /// is not a quaternion (caller falls back to readProperty).
    static func extractQuaternion(from value: TrideTypedValue) -> simd_quatf? {
        switch value {
        case .quat(let x, let y, let z, let w):
            return simd_quatf(ix: Float(x), iy: Float(y), iz: Float(z), r: Float(w))
        case .vec3(let x, let y, let z):
            warnOnceLegacyEulerRotation(nodeIndex: nil)
            return eulerXYZQuaternion(SIMD3(Float(x), Float(y), Float(z)))
        default:
            return nil
        }
    }

    /// Extract a vec3 from TrideTypedValue. Broadcasts scalar to (s,s,s)
    /// for uniform scale bindings.
    static func extractVec3(from value: TrideTypedValue) -> SIMD3<Float>? {
        switch value {
        case .vec3(let x, let y, let z):
            return SIMD3(Float(x), Float(y), Float(z))
        case .scalar(let v):
            let f = Float(v)
            return SIMD3(f, f, f)
        default:
            return nil
        }
    }

    /// Extract a linear RGB colour: the core's `color` (alpha dropped — the lerp's 1) or
    /// a bare vec3.
    static func extractRgb(from value: TrideTypedValue) -> SIMD3<Float>? {
        switch value {
        case .color(let r, let g, let b, _):
            return SIMD3(Float(r), Float(g), Float(b))
        case .vec3(let x, let y, let z):
            return SIMD3(Float(x), Float(y), Float(z))
        default:
            return nil
        }
    }

    /// Extract a scalar from TrideTypedValue.
    static func extractScalar(from value: TrideTypedValue) -> Double? {
        switch value {
        case .scalar(let v): return v
        default: return nil
        }
    }

    /// Extract a slug from a scene-domain enum mutation (`/scene/active_camera`,
    /// wire = JSON string). Accepts the typed string lane and the
    /// raw-JSON lane (a quoted string fragment); anything else → `nil`.
    static func extractString(from value: TrideTypedValue) -> String? {
        switch value {
        case .stringValue(let v):
            return v
        case .json(let raw):
            guard let data = raw.data(using: .utf8),
                  let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
            else { return nil }
            return parsed as? String
        default:
            return nil
        }
    }

    // MARK: - Value reading (FFI fallback)

    /// Upload the trunks the last dispatch deferred (their modules and pulse tables with them).
    /// `TrideKitScene` calls it before the next tick, and at once when no next tick is coming.
    /// Returns whether anything was uploaded.
    @discardableResult
    func flushDeferredTrunks(into bridge: FilamentBridge) -> Bool {
        guard !deferredTrunks.isEmpty else { return false }
        let pending = deferredTrunks
        deferredTrunks = []
        for (id, trunk) in pending {
            uploadTrunk(id: id, trunk: trunk, into: bridge)
            uploadGeneratorFlow(id: id, into: bridge)
        }
        // A mesh re-uploaded under a hidden host.
        if nodeVisibleOwn.values.contains(false) { applyNodeVisibility(to: bridge) }
        return true
    }

    private func uploadTrunk(id: String, trunk: UInt64?, into bridge: FilamentBridge) {
        // One event per trunk upload — with `dispatch` (one per tick) it turns a
        // profile's ms/s into ms per re-evaluation and per tick.
        signposts.emitEvent("trunk upload", "\(id, privacy: .public)")
        uploadGeneratedMesh(id: id, into: bridge)
        if let trunk { uploadedTrunkEpoch[id] = trunk }
    }

    /// `trunkEpoch` of a `/generators/{id}/geometry` announcement, `nil` when absent.
    static func trunkEpoch(of value: TrideTypedValue) -> UInt64? {
        guard case .json(let raw) = value,
              let object = try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any],
              let trunk = object["trunkEpoch"] as? NSNumber
        else { return nil }
        return trunk.uint64Value
    }

    private func readVec3(pointer: String) -> SIMD3<Float>? {
        // Manifest bindings sometimes write a scalar to a vec3 pointer
        // (e.g. uniform `lerp(1, 1.5, progress)` → `/nodes/3/scale`).
        // The Rust runtime stores the scalar verbatim — readNumberArray
        // returns nil for it. Fall back to readScalar and broadcast
        // (s, s, s) so uniform-scale bindings still apply.
        if let array = readNumberArray(pointer: pointer), array.count >= 3 {
            return SIMD3(array[0], array[1], array[2])
        }
        if let s = readScalar(pointer: pointer) {
            return SIMD3(s, s, s)
        }
        return nil
    }

    private func readScalar(pointer: String) -> Float? {
        guard let json = scene.readProperty(pointer: pointer),
              let data = json.data(using: .utf8),
              let raw = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else { return nil }
        if let n = raw as? NSNumber { return n.floatValue }
        return nil
    }

    /// Read a material's `baseColorFactor` pointer and return its alpha
    /// channel (4th element). Bindings often write only one component
    /// (`/materials/0/.../baseColorFactor/3`) — but `readProperty` on
    /// the parent pointer returns the full 4-vector, so we read once.
    private func readMaterialAlpha(materialIndex idx: Int) -> Float? {
        // glTF materials normally carry pbrMetallicRoughness; material
        // extensions may live elsewhere. We try the canonical
        // path first and fall back to the broader search.
        if let array = readNumberArray(
            pointer: "/materials/\(idx)/pbrMetallicRoughness/baseColorFactor"
        ), array.count >= 4 {
            return array[3]
        }
        if let array = readNumberArray(pointer: "/materials/\(idx)/baseColorFactor"),
           array.count >= 4 {
            return array[3]
        }
        return nil
    }

    /// Read a JSON Pointer through the Rust runtime and decode the
    /// returned JSON string into `[Float]`. Returns `nil` if the pointer
    /// is unset or the value is not a numeric array.
    private func readNumberArray(pointer: String) -> [Float]? {
        guard let json = scene.readProperty(pointer: pointer) else { return nil }
        guard let data = json.data(using: .utf8) else { return nil }
        guard let raw = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else { return nil }
        if let array = raw as? [NSNumber] {
            return array.map { $0.floatValue }
        }
        if let nested = raw as? [Any] {
            return nested.compactMap { ($0 as? NSNumber)?.floatValue }
        }
        return nil
    }

    // MARK: - glTF JSON walkers (statically reachable for tests)

    /// Walk nodes → mesh → primitives → material; produce
    /// `materialIndex → [nodeIndex]`. Index-based:
    /// each node is identified by its position in the glTF `nodes[]` array,
    /// not by name. Correctly handles duplicate node names.
    static func collectMaterialBindingsByIndex(
        fromNodesArray nodes: [Any],
        meshesArray meshes: [Any]
    ) -> [Int: [Int]] {
        var out: [Int: [Int]] = [:]
        for (nodeIndex, raw) in nodes.enumerated() {
            guard let node = raw as? [String: Any],
                  let meshIdx = (node["mesh"] as? NSNumber)?.intValue,
                  meshIdx >= 0,
                  meshIdx < meshes.count,
                  let mesh = meshes[meshIdx] as? [String: Any],
                  let primitives = mesh["primitives"] as? [[String: Any]]
            else { continue }
            for primitive in primitives {
                guard let materialIdx = (primitive["material"] as? NSNumber)?.intValue else { continue }
                out[materialIdx, default: []].append(nodeIndex)
            }
        }
        return out
    }


    /// Read a pointer through the runtime, expecting a JSON array.
    /// Returns `nil` if the pointer is missing or not an array.
    private func readJsonArray(pointer: String) -> [Any]? {
        guard let json = scene.readProperty(pointer: pointer),
              let data = json.data(using: .utf8),
              let raw = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else { return nil }
        return raw as? [Any]
    }
}

#endif  // canImport(TrideKitFilamentBridge): a host without the bridge (macOS) gets TrideCore only
