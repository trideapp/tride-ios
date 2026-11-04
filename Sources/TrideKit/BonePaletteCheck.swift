/// The bone palette check (the parity gate's palette line): each palette the core handed
/// (`TickResult.bonePalettes`) against the one gltfio's `AnimatorImpl::updateBoneMatrices` makes of
/// the host's own worlds — `inv(W_mesh)` by Gauss–Jordan in `Double`, `× W_joint` in `Double`,
/// narrowed to `Float`, `× IBM` in `Float` (Filament 1.77.1 `Animator.cpp`).
///
/// ## Public API
/// - `BonePaletteCheck.Result` — palettes and bones compared, the worst difference relative to the
///   largest element of the last product's factors (`f32(inv(W_mesh) × W_joint)` and the IBM), the
///   palettes whose mesh the view has no world for.
/// - `BonePaletteCheck.tolerance` — 1.9e-6: the core's host measure × 2.
/// - Inside TrideKit: `key(mesh:part:)`, `gltfioBone(meshWorld:jointWorld:ibm:)`,
///   `measure(palettes:skins:partJoints:world:)`.
///
/// ## Non-obvious constraints
/// - Filament hands no palette back: the host's side is recomputed from its TransformManager worlds
///   (`FilamentBridge.worldTransform(atIndex:)`), the mesh's and each joint's place.
/// - A joint the record lacks (`nil` row) is the identity bone, as the core makes it; an unsplit
///   palette's bone `j` is the skin's joint `j`, a skin part's the part's `joints[j]`.
/// - Cross-refs: `crates/tride-runtime-api/tests/skin_palette_gltfio.rs` (the same arithmetic and
///   bound, against gltfio's oracle).

import Foundation

#if canImport(TrideCore)
import TrideCore
#endif

public enum BonePaletteCheck {
    /// The worst difference allowed, relative to the largest element of the last product's factors.
    public static let tolerance = 1.9e-6

    public struct Result: Sendable, Equatable {
        public let palettes: Int
        public let bones: Int
        /// The worst `|core − host| / max|factors|` over every element of every bone.
        public let worst: Double
        /// Palettes whose mesh place the view has no world for (not compared).
        public let unplaced: Int

        public var passes: Bool { palettes > 0 && unplaced == 0 && worst <= BonePaletteCheck.tolerance }
    }

    /// A palette's key: the mesh node's row and the skin part.
    static func key(mesh: UInt32, part: UInt32) -> UInt64 {
        (UInt64(mesh) << 32) | UInt64(part)
    }

    /// gltfio's bone of `jointWorld` for a mesh at `meshWorld`, column-major, with the largest element
    /// of the last product's two factors.
    static func gltfioBone(meshWorld: [Float], jointWorld: [Float], ibm: [Float]) -> (bone: [Float], factor: Double) {
        let inverse = gaussJordan(meshWorld.map(Double.init))
        let global = multiply(inverse, jointWorld.map(Double.init)).map { Float($0) }
        let factor = (global + ibm).reduce(0.0) { max($0, Double(abs($1))) }
        return (multiply(global, ibm), factor)
    }

    /// Every palette against the host's: `skins` the record's, `partJoints` a cut part's joints by
    /// `key`, `world` a place's world matrix as the view composed it.
    static func measure(
        palettes: [TrideBonePalette], skins: [TrideRecordSkin], partJoints: [UInt64: [UInt32]],
        world: (Int) -> [Float]?
    ) -> Result {
        var compared = 0
        var bones = 0
        var worst = 0.0
        var unplaced = 0
        for palette in palettes {
            guard Int(palette.skin) < skins.count, let meshWorld = world(Int(palette.mesh)) else {
                unplaced += 1
                continue
            }
            let skin = skins[Int(palette.skin)]
            let core = floats(palette.bones)
            let ibms = floats(skin.inverseBindMatrices)
            let count = core.count / 16
            let joints = partJoints[key(mesh: palette.mesh, part: palette.part)] ?? (0..<UInt32(count)).map { $0 }
            compared += 1
            for j in 0..<min(count, joints.count) {
                let joint = Int(joints[j])
                var host = identity
                var factor = 1.0
                if joint < skin.joints.count, let row = skin.joints[joint], let jointWorld = world(Int(row)),
                   (joint + 1) * 16 <= ibms.count
                {
                    (host, factor) = gltfioBone(
                        meshWorld: meshWorld, jointWorld: jointWorld, ibm: Array(ibms[joint * 16..<(joint + 1) * 16]))
                }
                for e in 0..<16 {
                    worst = max(worst, abs(Double(core[j * 16 + e]) - Double(host[e])) / factor)
                }
                bones += 1
            }
        }
        return Result(palettes: compared, bones: bones, worst: worst, unplaced: unplaced)
    }

    private static let identity: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    /// Little-endian `f32` stream.
    private static func floats(_ data: Data) -> [Float] {
        data.withUnsafeBytes { raw in
            (0..<raw.count / 4).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }

    /// `TMat44 * TMat44` as Filament's `matrix::multiply`: each column `lhs * rhs[col]`, the sum
    /// taken row by row over `c`.
    private static func multiply<T: BinaryFloatingPoint>(_ a: [T], _ b: [T]) -> [T] {
        var out = [T](repeating: 0, count: 16)
        for col in 0..<4 {
            var result: [T] = [0, 0, 0, 0]
            for c in 0..<4 {
                for row in 0..<4 { result[row] += a[c * 4 + row] * b[col * 4 + c] }
            }
            for row in 0..<4 { out[col * 4 + row] = result[row] }
        }
        return out
    }

    /// `matrix::gaussJordanInverse` (`TMatHelpers.h`), `m[col][row]`.
    private static func gaussJordan(_ m: [Double]) -> [Double] {
        var src = (0..<4).map { c in Array(m[c * 4..<c * 4 + 4]) }
        var inverted = (0..<4).map { c in (0..<4).map { $0 == c ? 1.0 : 0.0 } }
        for i in 0..<4 {
            var swap = i
            var t = abs(src[i][i])
            for j in i + 1..<4 where abs(src[j][i]) > t {
                swap = j
                t = abs(src[j][i])
            }
            if swap != i {
                src.swapAt(i, swap)
                inverted.swapAt(i, swap)
            }
            let denom = src[i][i]
            for k in 0..<4 {
                src[i][k] /= denom
                inverted[i][k] /= denom
            }
            for j in 0..<4 where j != i {
                let t = src[j][i]
                for k in 0..<4 {
                    src[j][k] -= src[i][k] * t
                    inverted[j][k] -= inverted[i][k] * t
                }
            }
        }
        return inverted.flatMap { $0 }
    }
}
