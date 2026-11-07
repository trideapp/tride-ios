/// A scene built from the core's scene record: the one table the view addresses nodes,
/// lights and cameras through — one place per node row; a light or camera sits on the row of the
/// node that holds it.
///
/// ## Public API (inside TrideKit)
/// - `RecordScene(record:)` — `places` (for `FilamentBridge.loadScene(places:)`), `placeOf`
///   (node uid → row), `lightPlace` / `cameraPlace` / `placeById` (id → its node's row), `lights` /
///   `cameras`, `unplaced` (ids no node holds), `uidAt` (row → uid),
///   `drawnVisible` (row → the core's visibility to the renderer at load), `datablockNodes`,
///   `materials` (the record's palette, how the core draws each slug), `restWeights` (node uid →
///   morph weights at rest), `skins` (the record's skins), `skinOf` (node uid → its skin).
/// - `RecordScene.worlds(of:)` — the record's rest worlds composed in `Double` (checks).
/// - `RecordScene.drawnRequires`, `requiresNotice(_:)` — what of a record's `requires` iOS draws, and
///   the one notice for what it does not.
///
/// ## Non-obvious constraints
/// - A place is a row's number: the bridge takes it wherever a glTF node index is taken, so every
///   node mutation, generator host and focus target goes through this table, never a name.
/// - A light / camera is a node: its component sits on the row whose `light` /
///   `camera` names it, as gltfio puts one on its node; its parameters are the record's resolved
///   numbers (the record holds no pose).
/// - Every node with a datablock draws the core's mesh under the node's entity (`datablockNodes`);
///   a file mesh wears the record's `materials` (the dispatcher hands them to the bridge); a node
///   with a skin is bent by the core's palettes (`TickResult.bonePalettes`, by its row).

import Foundation
import simd

#if canImport(TrideCore)
import TrideCore
#endif

struct RecordScene {
    /// One place per node row: `parent` (a place), `translation`, `rotation` (`[x, y, z, w]`),
    /// `scale`.
    let places: [[String: Any]]
    /// Node uid → its row (its place).
    let placeOf: [String: Int]
    /// Row → node uid.
    let uidAt: [Int: String]
    /// Light id → the row of the node that holds it.
    let lightPlace: [String: Int]
    /// Camera id → the row of the node that holds it.
    let cameraPlace: [String: Int]
    /// The record's lights and cameras, parameters resolved by the core.
    let lights: [TrideRecordLight]
    let cameras: [TrideRecordCamera]
    /// Light / camera ids no node row names (`/lights/{id}`, `/cameras/{id}`): nothing to put them on.
    let unplaced: [String]

    /// Node uid, light id and camera id → the row it is drawn at (a node's own uid first).
    var placeById: [String: Int] {
        placeOf.merging(lightPlace) { node, _ in node }.merging(cameraPlace) { first, _ in first }
    }
    /// Row → visible to the renderer at load (its own flag and every ancestor's, the core's).
    let drawnVisible: [Int: Bool]
    /// Every node that draws a datablock, as the parametric nodes are listed.
    let datablockNodes: [TrideParametricNode]
    /// The record's materials: per slug, the form each family draws it in.
    let materials: [TrideRecordMaterial]
    /// Node uid → its morph weights at rest (the node's, else its mesh's); absent without any.
    let restWeights: [String: [Double]]
    /// The record's skins (`joints` as rows, `inverseBindMatrices`), palette `skin` order.
    let skins: [TrideRecordSkin]
    /// Node uid → its skin (a position in `skins`); absent for a node no skin bends.
    let skinOf: [String: Int]

    /// What of a record's `requires` iOS draws: skins by the core's palettes, morphs, lines, points
    /// and strips.
    static let drawnRequires: Set<String> = ["skins", "morphs", "lines", "points", "strips"]

    /// The scene drawn without what its `requires` names and iOS does not draw (`drawnRequires`);
    /// `nil` when it draws all of it.
    static func requiresNotice(_ requires: [String]) -> String? {
        let missing = requires.filter { !drawnRequires.contains($0) }
        guard !missing.isEmpty else { return nil }
        return "the scene requires \(missing.joined(separator: ", ")), which iOS does not draw yet: drawn without them"
    }

    init(record: TrideSceneRecord) {
        var places: [[String: Any]] = []
        var placeOf: [String: Int] = [:]
        var uidAt: [Int: String] = [:]
        var drawnVisible: [Int: Bool] = [:]
        var datablockNodes: [TrideParametricNode] = []
        var restWeights: [String: [Double]] = [:]
        var skinOf: [String: Int] = [:]
        var lightPlace: [String: Int] = [:]
        var cameraPlace: [String: Int] = [:]
        for (row, node) in record.nodes.enumerated() {
            if let light = node.light { lightPlace[light] = row }
            if let camera = node.camera { cameraPlace[camera] = row }
            if !node.weights.isEmpty { restWeights[node.uid] = node.weights }
            if let skin = node.skin, Int(skin) < record.skins.count { skinOf[node.uid] = Int(skin) }
            placeOf[node.uid] = row
            uidAt[row] = node.uid
            drawnVisible[row] = node.visible
            if let datablock = node.datablock {
                datablockNodes.append(TrideParametricNode(nodeUid: node.uid, datablockUid: datablock))
            }
            places.append(Self.place(
                parent: node.parent, translation: node.translation, rotation: node.rotation, scale: node.scale))
        }
        self.places = places
        self.placeOf = placeOf
        self.uidAt = uidAt
        self.lightPlace = lightPlace
        self.cameraPlace = cameraPlace
        self.lights = record.lights
        self.cameras = record.cameras
        self.unplaced = record.lights.filter { lightPlace[$0.id] == nil }.map { "/lights/\($0.id)" }
            + record.cameras.filter { cameraPlace[$0.id] == nil }.map { "/cameras/\($0.id)" }
        self.drawnVisible = drawnVisible
        self.datablockNodes = datablockNodes
        self.materials = record.materials
        self.restWeights = restWeights
        self.skins = record.skins
        self.skinOf = skinOf
    }

    private static func place(
        parent: UInt32?, translation: [Double], rotation: [Double], scale: [Double]
    ) -> [String: Any] {
        var place: [String: Any] = ["translation": translation, "rotation": rotation, "scale": scale]
        if let parent { place["parent"] = NSNumber(value: parent) }
        return place
    }

    /// Each place's world matrix (column-major, 16 numbers) composed from the record in `Double`:
    /// `parent world × T × R × S` (glTF §3.5). A place whose parent is no place hangs at the root, as
    /// the bridge hangs it.
    static func worlds(of places: [[String: Any]]) -> [[Double]] {
        var worlds: [simd_double4x4?] = Array(repeating: nil, count: places.count)
        func world(_ i: Int, depth: Int) -> simd_double4x4 {
            if let known = worlds[i] { return known }
            let place = places[i]
            let local = localMatrix(place)
            var parentWorld = matrix_identity_double4x4
            if let p = (place["parent"] as? NSNumber)?.intValue, p >= 0, p < places.count, p != i,
               depth < places.count
            {
                parentWorld = world(p, depth: depth + 1)
            }
            let out = parentWorld * local
            worlds[i] = out
            return out
        }
        return places.indices.map { i in
            let m = world(i, depth: 0)
            return (0..<4).flatMap { c in (0..<4).map { r in m[c][r] } }
        }
    }

    private static func localMatrix(_ place: [String: Any]) -> simd_double4x4 {
        let t = place["translation"] as? [Double] ?? [0, 0, 0]
        let q = place["rotation"] as? [Double] ?? [0, 0, 0, 1]
        let s = place["scale"] as? [Double] ?? [1, 1, 1]
        let rotation = simd_double4x4(simd_quatd(ix: q[0], iy: q[1], iz: q[2], r: q[3]))
        var m = rotation
        m.columns.0 *= s[0]
        m.columns.1 *= s[1]
        m.columns.2 *= s[2]
        m.columns.3 = simd_double4(t[0], t[1], t[2], 1)
        return m
    }
}
