/// The Metal-library build of a scene's node-material package, put next to the scene by the
/// `TrideMaterials` build-tool plugin.
///
/// ## Public API
/// - `PrecompiledMaterials.package(for:in:)` — the precompiled package for the bytes a `.trd` holds, or `nil`.
/// - `PrecompiledMaterials.url(for:in:)` — where that file is, on any platform (tests, diagnostics).
///
/// ## Non-obvious constraints
/// - Keyed by the sha256 of the package bytes as the `.trd` holds them (`tride-metal-<hex>.filamat`):
///   the scene file stays as authored; one structure shared by several scenes is one file.
/// - The plugin writes the files into the bundle of the target it is applied to: the app's
///   (`Bundle.main`) or a package target's (`Bundle.module`, the target's resource bundle). The scene's
///   own bundle (`TrideKitScene(trdBytes:bundle:)`) is asked first, then `Bundle.main`.
/// - Device only — the simulator cannot load iphoneos libraries and keeps the package's Metal source.
/// - Without the plugin (or for a scene the app downloads) nothing is found and Filament compiles the
///   source on first use, as before.

import CryptoKit
import Foundation

enum PrecompiledMaterials {
    /// Where the plugin's side files are looked for: the scene's bundle, then the app's; once each.
    static func bundles(scene: Bundle?) -> [Bundle] {
        guard let scene, scene.bundleURL != Bundle.main.bundleURL else { return [.main] }
        return [scene, .main]
    }

    /// The first of `bundles` holding the precompiled build of `package`, on any platform.
    static func url(for package: Data, in bundles: [Bundle]) -> URL? {
        let key = SHA256.hash(data: package).map { String(format: "%02x", $0) }.joined()
        for bundle in bundles {
            if let url = bundle.url(forResource: "tride-metal-\(key)", withExtension: "filamat") { return url }
        }
        return nil
    }

    /// `throws` only when a precompiled file exists but cannot be read.
    static func package(for package: Data, in bundles: [Bundle]) throws -> Data? {
        #if targetEnvironment(simulator)
        return nil
        #else
        guard let url = url(for: package, in: bundles) else { return nil }
        return try Data(contentsOf: url)
        #endif
    }
}
