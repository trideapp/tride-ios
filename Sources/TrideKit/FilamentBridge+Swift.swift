/// Swift convenience overloads bridging `simd` types onto ObjC `FilamentBridge` entry points.
///
/// ## Public API
/// - Internal `FilamentBridge` extensions — `simd_quatf`/`SIMD3<Float>` adapters for rotation, scale, translation.
/// - `FilamentBridge.eulerXYZRotationMatrix` — Euler XYZ radians → column-major `simd_float3x3`
///   (the IBL rotation boundary conversion); + `eulerXYZ`-based IBL overloads.
///
/// ## Non-obvious constraints
/// - Zero runtime cost — forwards to ObjC selectors; `simd_quatf.vector` layout (`ix, iy, iz, r`)
///   maps one-to-one to the bridge's `simd_float4 = (x, y, z, w)` quaternion packing,
///   the canonical rotation (4-component `[qx, qy, qz, qw]`, unit-norm).
/// - IBL rotation is resolved Euler→matrix HERE at the bridge boundary (the core keeps Euler),
///   from the core's composition (`PointerDispatcher.eulerXYZQuaternion` → `eulerToQuat`), like the
///   light path's Euler conversion.

import Foundation
import simd

#if canImport(TrideKitFilamentBridge)
import TrideKitFilamentBridge

extension FilamentBridge {

    /// `simd_quatf` overload (`(ix, iy, iz, real)` layout), converted to the bridge's
    /// `simd_float4 = (x, y, z, w)`.
    func setNodeRotation(_ nodeName: String, quaternion: simd_quatf) {
        let v = quaternion.vector  // (ix, iy, iz, r) == (x, y, z, w)
        setNodeRotation(nodeName, quaternion: v)
    }

    /// Canonical JSON Pointer variant: `/nodes/{N}/rotation`.
    func setNodeRotation(atIndex nodeIndex: Int, quaternion: simd_quatf) {
        let v = quaternion.vector  // (ix, iy, iz, r) == (x, y, z, w)
        setNodeRotation(atIndex: nodeIndex, quaternion: v)
    }

    /// `SIMD3<Float>` overload.
    func setNodeScale(_ nodeName: String, scale: SIMD3<Float>) {
        setNodeScale(nodeName, x: scale.x, y: scale.y, z: scale.z)
    }

    /// Canonical JSON Pointer variant: `/nodes/{N}/scale`.
    func setNodeScale(atIndex nodeIndex: Int, scale: SIMD3<Float>) {
        setNodeScale(atIndex: nodeIndex, x: scale.x, y: scale.y, z: scale.z)
    }

    /// SIMD3 convenience for `/nodes/{name}/translation`.
    func setNodeTranslation(_ nodeName: String, translation: SIMD3<Float>) {
        setNodeTranslation(nodeName, x: translation.x, y: translation.y, z: translation.z)
    }

    /// Canonical JSON Pointer variant: `/nodes/{N}/translation`.
    func setNodeTranslation(atIndex nodeIndex: Int, translation: SIMD3<Float>) {
        setNodeTranslation(atIndex: nodeIndex, x: translation.x, y: translation.y, z: translation.z)
    }

    /// OPT-3 coalesced TRS: accepts `simd_quatf` for rotation (converted
    /// to `simd_float4` via `.vector`), `SIMD3<Float>` for T/S.
    func setNodeTRS(
        _ nodeName: String,
        translation: SIMD3<Float>,
        rotation: simd_quatf,
        scale: SIMD3<Float>,
        mask: FilamentBridgeTRSMask
    ) {
        setNodeTRS(nodeName, translation: translation, rotation: rotation.vector, scale: scale, mask: mask)
    }

    /// Build a column-major rotation matrix from Euler XYZ angles in radians,
    /// resolving the orientation at the bridge boundary. Order is
    /// `Q = Qx · Qy · Qz` — the same body-frame XYZ convention the light-glyph
    /// / node-rotation paths use (`PointerDispatcher.eulerXYZQuaternion`),
    /// matching the editor preview. Pure and
    /// Filament-free so it is unit-testable without a live engine.
    static func eulerXYZRotationMatrix(_ euler: SIMD3<Float>) -> simd_float3x3 {
        simd_float3x3(PointerDispatcher.eulerXYZQuaternion(euler))
    }

    /// Apply an HDR environment whose orientation is given as Euler XYZ radians
    /// (`environment_settings.rotation`); converts to the column-major matrix
    /// the ObjC surface consumes. `skyboxKtx == nil` → IBL-only (background not
    /// visible).
    func setIndirectLight(
        _ iblKtx: Data,
        skybox skyboxKtx: Data?,
        intensity: Float,
        rotationEulerXYZ euler: SIMD3<Float>
    ) {
        setIndirectLight(
            iblKtx,
            skybox: skyboxKtx,
            intensity: intensity,
            rotation: Self.eulerXYZRotationMatrix(euler))
    }

    /// Update the live IBL orientation from Euler XYZ radians.
    func setIndirectLightRotation(eulerXYZ euler: SIMD3<Float>) {
        setIndirectLightRotation(Self.eulerXYZRotationMatrix(euler))
    }
}

/// The bridge's errors as a Swift enum instead of a raw `NSError` — e.g. to match the error
/// `FilamentMetalView.onLoadError` receives.
public enum TrideFilamentError: Error, Equatable {
    case bridgeDisposed
    case assetParseFailed
    case resourceLoadFailed
    case assetAlreadyLoaded
    case unknown(code: Int, description: String)

    /// Maps the `NSError` the ObjC bridge throws.
    public init(_ error: Error) {
        let nsErr = error as NSError
        guard nsErr.domain == FilamentBridgeErrorDomain else {
            self = .unknown(code: nsErr.code, description: nsErr.localizedDescription)
            return
        }
        switch nsErr.code {
        case Int(FilamentBridgeErrorCode.disposed.rawValue):
            self = .bridgeDisposed
        case Int(FilamentBridgeErrorCode.assetParseFailed.rawValue):
            self = .assetParseFailed
        case Int(FilamentBridgeErrorCode.resourceLoadFailed.rawValue):
            self = .resourceLoadFailed
        case Int(FilamentBridgeErrorCode.assetAlreadyLoaded.rawValue):
            self = .assetAlreadyLoaded
        default:
            self = .unknown(code: nsErr.code, description: nsErr.localizedDescription)
        }
    }
}

#endif  // canImport(TrideKitFilamentBridge): a host without the bridge (macOS) gets TrideCore only
