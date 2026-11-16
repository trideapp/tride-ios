/// The core's environment frame (`TrideScene.environmentFrame()`) in the bridge's shape, number for number.
///
/// ## Public API
/// - Internal `FilamentBridgeEnvironmentValues.init(_:)` — `TrideEnvironmentFrame` → the C struct that
///   `FilamentBridge.applyEnvironment(_:lightKtx:skyKtx:)` / `updateEnvironment(_:)` take.
/// - Internal `TrideEnvironmentFrame.lightMapSource`, `.skyMapSource` — the maps whose KTX the host reads.
///
/// ## Non-obvious constraints
/// - No formula here (one core for every host): matrices are the core's column-major 3×3, stacks
///   its premultiplied RGBA; a short or missing array becomes identity / zero, never a computed value.
/// - Light kind: `"color"` / `"map"`; anything else (`"none"`) lights nothing.
///
/// ## Cross-refs
/// - `crates/tride-ffi` `TrideEnvironmentFrame`; `FilamentBridge.h` (Environment); `TrideKitScene`
///   (load), `PointerDispatcher` (after a write under `/environments/…`).

import Foundation
import simd
import TrideCore

#if canImport(TrideKitFilamentBridge)
import TrideKitFilamentBridge

extension FilamentBridgeEnvironmentValues {
    init(_ frame: TrideEnvironmentFrame) {
        self.init()
        switch frame.light.kind {
        case "color": light = .color
        case "map": light = .map
        default: light = .none
        }
        lightColor = Self.float3(frame.light.rgb)
        lux = Float(frame.light.lux)
        lightRotation = Self.matrix(frame.light.rotation)
        sky = Self.stack(frame.sky)
        glass = Self.stack(frame.glass)
        transparent = frame.transparent
    }

    /// Column-major 3×3 as the core wrote it; identity when it is not nine numbers.
    static func matrix(_ m: [Double]) -> simd_float3x3 {
        guard m.count == 9 else { return matrix_identity_float3x3 }
        let f = m.map(Float.init)
        return simd_float3x3(columns: (
            SIMD3(f[0], f[1], f[2]), SIMD3(f[3], f[4], f[5]), SIMD3(f[6], f[7], f[8])))
    }

    static func float3(_ v: [Double]) -> simd_float3 {
        v.count >= 3 ? SIMD3(Float(v[0]), Float(v[1]), Float(v[2])) : .zero
    }

    static func float4(_ v: [Double]) -> simd_float4 {
        v.count >= 4 ? SIMD4(Float(v[0]), Float(v[1]), Float(v[2]), Float(v[3])) : .zero
    }

    static func stack(_ s: TrideSkyStack) -> FilamentBridgeSkyStack {
        FilamentBridgeSkyStack(
            over: float4(s.over), under: float4(s.under),
            mapScale: Float(s.map?.scale ?? 0), mapAlpha: Float(s.map?.alpha ?? 0),
            mapRotation: s.map.map { matrix($0.rotation) } ?? matrix_identity_float3x3)
    }
}
#endif

extension TrideEnvironmentFrame {
    /// The map that lights the scene; `nil` for a colour or nothing.
    var lightMapSource: TrideEnvironmentSourceRef? { light.kind == "map" ? light.source : nil }

    /// The shown map (the same in `sky` and `glass`); `nil` when no map is shown.
    var skyMapSource: TrideEnvironmentSourceRef? { sky.map?.source ?? glass.map?.source }
}
