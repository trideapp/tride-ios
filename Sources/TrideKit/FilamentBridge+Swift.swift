/// Swift convenience overloads bridging `simd` types onto ObjC `FilamentBridge` entry points.
///
/// ## Public API
/// - Internal `FilamentBridge` extensions — `simd_quatf`/`SIMD3<Float>` adapters for a place's
///   rotation, scale, translation (by its number, `/nodes/{N}` after the dispatcher's rewrite).
///
/// ## Non-obvious constraints
/// - Zero runtime cost — forwards to ObjC selectors; `simd_quatf.vector` layout (`ix, iy, iz, r`)
///   maps one-to-one to the bridge's `simd_float4 = (x, y, z, w)` quaternion packing,
///   the canonical rotation (4-component `[qx, qy, qz, qw]`, unit-norm).
/// - The environment's rotations are the core's matrices (`EnvironmentFrameValues.swift`).

import Foundation
import simd

#if canImport(TrideKitFilamentBridge)
import TrideKitFilamentBridge

extension FilamentBridge {

    /// `simd_quatf` overload (`(ix, iy, iz, real)` layout), converted to the bridge's
    /// `simd_float4 = (x, y, z, w)`: `/nodes/{N}/rotation`.
    func setNodeRotation(atIndex nodeIndex: Int, quaternion: simd_quatf) {
        let v = quaternion.vector  // (ix, iy, iz, r) == (x, y, z, w)
        setNodeRotation(atIndex: nodeIndex, quaternion: v)
    }

    /// `SIMD3<Float>` overload: `/nodes/{N}/scale`.
    func setNodeScale(atIndex nodeIndex: Int, scale: SIMD3<Float>) {
        setNodeScale(atIndex: nodeIndex, x: scale.x, y: scale.y, z: scale.z)
    }

    /// SIMD3 convenience: `/nodes/{N}/translation`.
    func setNodeTranslation(atIndex nodeIndex: Int, translation: SIMD3<Float>) {
        setNodeTranslation(atIndex: nodeIndex, x: translation.x, y: translation.y, z: translation.z)
    }
}

/// The bridge's errors as a Swift enum instead of a raw `NSError` — e.g. to match the error
/// `FilamentMetalView.onLoadError` receives.
public enum TrideFilamentError: Error, Equatable {
    case bridgeDisposed
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
