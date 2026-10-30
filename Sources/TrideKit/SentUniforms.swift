/// What the view last handed the bridge of each node material's uniforms — the bridge's half of
/// the core's changed-only rule: only uniforms whose lanes moved are sent again.
///
/// ## Public API
/// None: `SentUniforms` is internal to TrideKit (`PointerDispatcher`).
///
/// ## Non-obvious constraints
/// - The core hands a node material's whole table on every pull (`materialGraphValues`), however
///   few inputs a tick wrote; the bridge keeps the table and an instance made later starts from it,
///   so a send carries the moved uniforms only and the others keep theirs.
/// - Keyed by palette slug and family. `removeAll()` at every bind (a new bridge, a scene reload):
///   the first push after it is the whole table, so nothing stale survives a re-bind.
/// - Lanes compare exactly (`[Double]` ==): a value the core recomputed to the same bits is not
///   sent, any other is. The same rule on Android (`SentUniforms.kt`) and the web
///   (`filament-material-graphs.ts`).

#if canImport(TrideCore)
import TrideCore
#endif

struct SentUniforms {
    private var sent: [String: [String: [Double]]] = [:]

    /// The uniforms of `values` whose lanes differ from the ones last handed over for `slug` on
    /// `family`; they become the last.
    mutating func changes(
        slug: String, family: String, values: [TrideMaterialGraphValue]
    ) -> [TrideMaterialGraphValue] {
        let key = "\(slug)\u{0}\(family)"
        var held = sent[key] ?? [:]
        let moved = values.filter { held[$0.name] != $0.values }
        for value in moved { held[value.name] = value.values }
        sent[key] = held
        return moved
    }

    mutating func removeAll() {
        sent = [:]
    }
}
