/// The Metal-library build of a scene's node-material package, put in the app bundle by the
/// `TrideMaterials` build-tool plugin.
///
/// ## Public API
/// - `PrecompiledMaterials.package(for:)` — the precompiled package for the bytes a `.trd` holds, or `nil`.
///
/// ## Non-obvious constraints
/// - Keyed by the sha256 of the package bytes as the `.trd` holds them (`tride-metal-<hex>.filamat` in
///   `Bundle.main`): the scene file stays as authored; one structure shared by several scenes is one file.
/// - Device only — the simulator cannot load iphoneos libraries and keeps the package's Metal source.
/// - Without the plugin (or for a scene the app downloads) nothing is found and Filament compiles the
///   source on first use, as before.

import CryptoKit
import Foundation

enum PrecompiledMaterials {
    /// `throws` only when a precompiled file exists but cannot be read.
    static func package(for package: Data, in bundle: Bundle = .main) throws -> Data? {
        #if targetEnvironment(simulator)
        return nil
        #else
        let key = SHA256.hash(data: package).map { String(format: "%02x", $0) }.joined()
        guard let url = bundle.url(forResource: "tride-metal-\(key)", withExtension: "filamat") else {
            return nil
        }
        return try Data(contentsOf: url)
        #endif
    }
}
