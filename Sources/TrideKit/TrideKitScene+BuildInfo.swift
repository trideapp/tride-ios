/// Which Tride runtime an app carries — the first thing support asks.
///
/// Public API: `TrideKitScene.buildInfo` → `TrideBuildInfo` (`version` — the release every Tride
/// stream carries, `commit` — the commit the runtime was built from, `formatVersion` — the `.trd`
/// format it reads).
///
/// Non-obvious constraints:
///   - Static: no scene has to be loaded — the runtime binary answers.
///   - `nonisolated`: the core call is synchronous and thread-safe, so any thread may ask.
///   - `commit` is empty for a runtime built outside the artifact pipeline.
import TrideCore

#if canImport(TrideKitFilamentBridge)
@available(iOS 17.0, macCatalyst 17.0, *)
public extension TrideKitScene {
    /// The runtime inside this app: release (`0.5.0`), commit, `.trd` format.
    nonisolated static var buildInfo: TrideKitBuildInfo { TrideKitBuildInfo(TrideCore.buildInfo()) }
}
#endif
