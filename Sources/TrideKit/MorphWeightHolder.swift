/// The morph weight vectors a loaded scene holds, one per node place (its record row).
///
/// A lane absent from a batch says nothing: the runtime sends only values that changed, and lanes
/// written by different sources at different times (an animation track, a binding) must survive each
/// other, so the vector is held, not rebuilt from the batch. A node's vector starts at its weights at
/// rest (the record row's `weights`, the core's `morph::rest_weights`), the same weights its mesh is
/// uploaded with, so a lane nothing animates keeps them. A lane at or past the node's target count is
/// rejected, so the vector handed to Filament is never longer than the renderable's targets.
///
/// ## Public API
/// - `fold(place:count:rest:lanes:)`: merge a batch's lanes into the place's vector, the whole vector back.
/// - `held(place:)`: the vector held for the place, `nil` before any write; `reset()`: a scene attach.
///
/// ## Cross-refs
/// - Android twin: `tridekit-android/.../MorphWeightHolder.kt`, the same rule.
struct MorphWeightHolder {
    struct Fold: Equatable {
        let vector: [Float]
        let rejected: [Int]
    }

    private var held: [Int: [Float]] = [:]

    /// Merge `lanes` into the vector held for `place`, `count` lanes long; returns the whole vector. A
    /// first write starts from `rest` (cut or padded with zeros to `count`; `nil` = all zeros); a new
    /// target count keeps the lanes held and takes the added ones from `rest`.
    mutating func fold(place: Int, count: Int, rest: [Double]?, lanes: [Int: Float]) -> Fold {
        let count = max(count, 0)
        var vector = held[place].flatMap { $0.count == count ? $0 : nil }
            ?? Self.seed(count: count, rest: rest, before: held[place])
        var rejected: [Int] = []
        for (lane, value) in lanes {
            if lane >= 0, lane < count { vector[lane] = value } else { rejected.append(lane) }
        }
        held[place] = vector
        return Fold(vector: vector, rejected: rejected.sorted())
    }

    /// The vector held for the place, `nil` before any write.
    func held(place: Int) -> [Float]? { held[place] }

    /// Forget every held vector (a scene attach): the next writes start from that record's rest.
    mutating func reset() { held = [:] }

    private static func seed(count: Int, rest: [Double]?, before: [Float]?) -> [Float] {
        var vector = (0..<count).map { lane in Float(rest.flatMap { lane < $0.count ? $0[lane] : nil } ?? 0) }
        for (lane, value) in (before ?? []).prefix(count).enumerated() { vector[lane] = value }
        return vector
    }
}
