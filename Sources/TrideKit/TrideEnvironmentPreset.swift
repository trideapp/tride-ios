/// Resolves baked HDR-environment presets bundled in `Bundle.module` to raw KTX1 bytes.
///
/// ## API (internal to TrideKit)
/// - `TrideEnvironmentPreset` — the six cmgen-baked presets; resource base-names.
/// - `TrideEnvironmentPresetLoader.iblData` / `.skyboxData` — KTX1 blobs, `nil` when the resource is absent.
///
/// ## Non-obvious constraints
/// - Consumer: `TrideKitScene.applyEnvironment(to:)` → `FilamentBridge`
///   `setIndirectLight(_:skybox:intensity:rotation:)`, which builds the
///   `IndirectLight` / `Skybox` via the KTX1 helpers in `FilamentBridge.mm`.
/// - Diffuse irradiance SH is embedded in `<name>_ibl.ktx` metadata, so no companion `_sh.txt` is bundled.
/// - `Resources/ibl/` is a copy of the baked presets with no sync step, so it can lag a re-bake.

import Foundation

/// One of the six baked HDR-environment presets shipped with TrideKit
/// (Filament `cmgen` KTX1).
enum TrideEnvironmentPreset: String, CaseIterable, Sendable {
    case indoor
    case night
    case outdoor
    case overcast
    case studio
    case sunset

    /// Resource base-name of the prefiltered specular + embedded-SH IBL cubemap.
    var iblResourceName: String { "\(rawValue)_ibl" }

    /// Resource base-name of the background skybox cubemap.
    var skyboxResourceName: String { "\(rawValue)_skybox" }
}

/// Loads the bundled preset KTX1 blobs from `Bundle.module`. Returns `nil`
/// when the resource is absent (e.g. a host that strips the resource bundle).
enum TrideEnvironmentPresetLoader {

    /// Raw KTX1 bytes of the preset IBL cubemap (specular reflections + SH irradiance).
    static func iblData(_ preset: TrideEnvironmentPreset) -> Data? {
        ktxData(named: preset.iblResourceName)
    }

    /// Raw KTX1 bytes of the preset skybox cubemap (background environment).
    static func skyboxData(_ preset: TrideEnvironmentPreset) -> Data? {
        ktxData(named: preset.skyboxResourceName)
    }

    private static func ktxData(named name: String) -> Data? {
        guard let url = Bundle.module.url(
            forResource: name,
            withExtension: "ktx",
            subdirectory: "ibl"
        ) else {
            return nil
        }
        return try? Data(contentsOf: url)
    }
}
