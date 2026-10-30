/// Projects the FFI-normalized `TrideRenderSettings` onto the `FilamentBridge.applyRenderSettings(_:)` dictionary contract.
///
/// ## Public API
/// - `RenderSettingsDictionary.make(_:)` — `TrideRenderSettings` → `[String: Any]` (camelCase keys = Rust
///   `RenderSettings` field names; enums as the schema's wire strings).
///
/// ## Non-obvious constraints
/// - **No host-side defaults**: every value is already resolved by `tride-schema`
///   (`render_settings()` returns the normalized form), so this is a pure shape projection — the
///   `dictFloat(...)` fallbacks inside the bridge never fire for an authored manifest.
/// - `qualityPreset` is an editor setting and intentionally not projected.
/// - Called from `TrideKitScene.attachBridge`; the editor viewport applies the same settings.

import Foundation

#if canImport(TrideCore)
import TrideCore
#endif

@available(iOS 17.0, macCatalyst 17.0, *)
enum RenderSettingsDictionary {
    static func make(_ s: TrideRenderSettings) -> [String: Any] {
        [
            "antiAliasing": antiAliasing(s.antiAliasing),
            "dithering": dithering(s.dithering),
            "postProcessing": s.postProcessing,
            "shadowingEnabled": s.shadowingEnabled,
            "shadowType": shadowType(s.shadowType),
            "msaa": ["enabled": s.msaa.enabled, "sampleCount": Int(s.msaa.sampleCount)],
            "taa": [
                "enabled": s.taa.enabled,
                "feedback": s.taa.feedback,
                "filterWidth": s.taa.filterWidth,
                "sharpness": s.taa.sharpness,
            ],
            "ssao": [
                "enabled": s.ssao.enabled,
                "radius": s.ssao.radius,
                "power": s.ssao.power,
                "bias": s.ssao.bias,
                "intensity": s.ssao.intensity,
                "quality": quality(s.ssao.quality),
                "resolution": s.ssao.resolution,
            ],
            "bloom": [
                "enabled": s.bloom.enabled,
                "strength": s.bloom.strength,
                "resolution": Int(s.bloom.resolution),
                "levels": Int(s.bloom.levels),
                "threshold": s.bloom.threshold,
                "highlight": s.bloom.highlight,
                "quality": quality(s.bloom.quality),
                "lensFlare": s.bloom.lensFlare,
            ],
            "fog": [
                "enabled": s.fog.enabled,
                "distance": s.fog.distance,
                "maximumOpacity": s.fog.maximumOpacity,
                "height": s.fog.height,
                "heightFalloff": s.fog.heightFalloff,
                "density": s.fog.density,
                "fogColorFromIbl": s.fog.fogColorFromIbl,
                "color": [s.fog.colorR, s.fog.colorG, s.fog.colorB],
            ],
            "vignette": [
                "enabled": s.vignette.enabled,
                "midPoint": s.vignette.midPoint,
                "roundness": s.vignette.roundness,
                "feather": s.vignette.feather,
                "color": s.vignette.color,
            ],
            "dynamicResolution": [
                "enabled": s.dynamicResolution.enabled,
                "minScale": s.dynamicResolution.minScale,
                "maxScale": s.dynamicResolution.maxScale,
                "quality": quality(s.dynamicResolution.quality),
            ],
            "depthOfField": [
                "enabled": s.depthOfField.enabled,
                "cocScale": s.depthOfField.cocScale,
                "cocAspectRatio": s.depthOfField.cocAspectRatio,
                "maxApertureDiameter": s.depthOfField.maxApertureDiameter,
                "filter": dofFilter(s.depthOfField.filter),
                "nativeResolution": s.depthOfField.nativeResolution,
                "maxForegroundCoc": Int(s.depthOfField.maxForegroundCoc),
                "maxBackgroundCoc": Int(s.depthOfField.maxBackgroundCoc),
                "foregroundRingCount": Int(s.depthOfField.foregroundRingCount),
                "backgroundRingCount": Int(s.depthOfField.backgroundRingCount),
                "fastGatherRingCount": Int(s.depthOfField.fastGatherRingCount),
            ],
            "screenSpaceReflections": [
                "enabled": s.screenSpaceReflections.enabled,
                "thickness": s.screenSpaceReflections.thickness,
                "bias": s.screenSpaceReflections.bias,
                "maxDistance": s.screenSpaceReflections.maxDistance,
                "stride": s.screenSpaceReflections.stride,
            ],
            "vsmShadowOptions": [
                "anisotropy": Int(s.vsmShadowOptions.anisotropy),
                "mipmapping": s.vsmShadowOptions.mipmapping,
                "msaaSamples": Int(s.vsmShadowOptions.msaaSamples),
                "highPrecision": s.vsmShadowOptions.highPrecision,
                "minVarianceScale": s.vsmShadowOptions.minVarianceScale,
                "lightBleedReduction": s.vsmShadowOptions.lightBleedReduction,
            ],
            "softShadowOptions": [
                "penumbraScale": s.softShadowOptions.penumbraScale,
                "penumbraRatioScale": s.softShadowOptions.penumbraRatioScale,
            ],
            "toneMapping": [
                "operator": toneMapping(s.toneMapping.operator),
                // The Filament tone mapper the core builds for the operator.
                "mapper": toneMapper(toneCurve(operator: s.toneMapping.operator)),
                "exposure": s.toneMapping.exposure,
                "luminanceScaling": s.toneMapping.luminanceScaling,
                "gamutMapping": s.toneMapping.gamutMapping,
                "saturation": s.toneMapping.saturation,
            ],
            "pixelation": ["enabled": s.pixelation.enabled, "size": s.pixelation.size],
        ]
    }

    static func antiAliasing(_ v: TrideAntiAliasing) -> String {
        switch v {
        case .none: return "NONE"
        case .fxaa: return "FXAA"
        }
    }

    static func dithering(_ v: TrideDithering) -> String {
        switch v {
        case .none: return "NONE"
        case .temporal: return "TEMPORAL"
        }
    }

    static func shadowType(_ v: TrideShadowType) -> String {
        switch v {
        case .pcf: return "PCF"
        case .vsm: return "VSM"
        case .dpcf: return "DPCF"
        case .pcss: return "PCSS"
        case .pcfD: return "PCFd"
        }
    }

    static func toneMapping(_ v: TrideToneMapping) -> String {
        switch v {
        case .linear: return "LINEAR"
        case .acesLegacy: return "ACES_LEGACY"
        case .aces: return "ACES"
        case .filmic: return "FILMIC"
        case .displayRange: return "DISPLAY_RANGE"
        case .acesNeutral: return "ACES_NEUTRAL"
        case .pbrNeutral: return "PBR_NEUTRAL"
        case .agxPunchy: return "AGX_PUNCHY"
        }
    }

    /// The core's tone mapper as the bridge takes it: `kind`, and the generic curve's parameters.
    static func toneMapper(_ m: TrideToneMapper) -> [String: Any] {
        switch m {
        case .linear: return ["kind": "linear"]
        case .acesLegacy: return ["kind": "acesLegacy"]
        case .aces: return ["kind": "aces"]
        case .filmic: return ["kind": "filmic"]
        case .displayRange: return ["kind": "displayRange"]
        case let .generic(contrast, midGrayIn, midGrayOut, hdrMax):
            return [
                "kind": "generic", "contrast": contrast, "midGrayIn": midGrayIn,
                "midGrayOut": midGrayOut, "hdrMax": hdrMax,
            ]
        case .pbrNeutral: return ["kind": "pbrNeutral"]
        case .agxPunchy: return ["kind": "agxPunchy"]
        }
    }

    static func quality(_ v: TrideQualityLevel) -> String {
        switch v {
        case .low: return "LOW"
        case .medium: return "MEDIUM"
        case .high: return "HIGH"
        case .ultra: return "ULTRA"
        }
    }

    static func dofFilter(_ v: TrideDofFilter) -> String {
        switch v {
        case .none: return "NONE"
        case .unused: return "UNUSED"
        case .median: return "MEDIAN"
        }
    }
}
