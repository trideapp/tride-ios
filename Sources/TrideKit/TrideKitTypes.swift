/// The values TrideKit hands to the host, where the runtime's generated types would be awkward to use.
///
/// ## Public API
/// - `TrideKitBinding` / `TrideKitBindingKind` — what the scene declares (`availableBindings()`).
/// - `TrideKitInputsReport` / `TrideKitInputRefusal` — what `setInputs(json:strict:)` did.
/// - `TrideKitHit` — a node under a point (`pick(at:)`).
/// - `TrideKitBuildInfo` — which runtime the app carries (`TrideKitScene.buildInfo`).
/// - `TrideKitAdvance` — where a host script stopped (`advance(scriptJSON:)`).
/// - `TrideKitDiagnostic` — a problem the scene reports to the host (`subscribeDiagnostics`).
///
/// ## Non-obvious constraints
/// - Values themselves stay the runtime's `TrideValue` (`bool` / `number` / `string`, and `vec3` for any
///   vector: a 3-vector, a rotation `[x, y, z, w]` or a colour): it is already a
///   plain value type, and TrideKit and the runtime always ship as one version.
/// - The shapes match TrideKit for Android one to one.

import Foundation

#if canImport(TrideCore)
import TrideCore
#endif

/// `TrideValue` only holds values (`Bool`, `Double`, `String`, `[Float]`), so it crosses threads safely.
extension TrideValue: @unchecked @retroactive Sendable {}

/// The kind of a binding the scene declares.
public enum TrideKitBindingKind: String, Sendable, CaseIterable {
    case number, bool, `enum`, vec3, quat, color, trigger
    /// The app's picture: set with `TrideKitScene.setImage`, read with `imageInput`;
    /// `getBinding` gives its status (`"empty"` / `"loading"` / `"loaded"` / `"failed"`).
    case image
    /// Free text: set with `setBinding(name, .string(…))` or in a `setInputs` batch, read with
    /// `getBinding`. The scene checks its length and line breaks on every write.
    case string
    /// A kind this TrideKit does not know yet (a scene from a newer editor).
    case other
}

/// A value or trigger the scene declares.
public struct TrideKitBinding: Sendable, Equatable, CustomStringConvertible {
    /// The binding's name — what `setBinding` / `trigger` / `getBinding` take.
    public let name: String
    public let kind: TrideKitBindingKind
    /// The variants of an `enum` binding; empty for other kinds.
    public let options: [String]
    /// `true` when the scene computes the value itself (a formula, or a node of its bindings graph):
    /// the host reads it, never writes.
    public let computed: Bool

    public init(name: String, kind: TrideKitBindingKind, options: [String] = [], computed: Bool = false) {
        self.name = name
        self.kind = kind
        self.options = options
        self.computed = computed
    }

    init(_ descriptor: TrideBindingDescriptor) {
        self.init(
            name: descriptor.name,
            kind: TrideKitBindingKind(rawValue: descriptor.kind) ?? .other,
            options: descriptor.options,
            computed: descriptor.computed != nil
        )
    }

    public var description: String {
        let extra = options.isEmpty ? "" : " [\(options.joined(separator: ", "))]"
        return "\(name): \(kind.rawValue)\(extra)\(computed ? " (computed)" : "")"
    }
}

/// An input a batch refused (`TrideKitInputsReport.rejected`): the scene never cuts or clamps a value.
public struct TrideKitInputRefusal: Sendable, Equatable, Decodable, CustomStringConvertible {
    /// The input's name.
    public let name: String
    /// Why: `kind`, `variant`, `range`, `length`, `newline`, `computed`, `bound`, `image`, `trigger`.
    public let code: String
    public let message: String

    public init(name: String, code: String, message: String) {
        self.name = name
        self.code = code
        self.message = message
    }

    public var description: String { "\(name) (\(code)): \(message)" }
}

/// What `TrideKitScene.setInputs(json:strict:)` did.
public struct TrideKitInputsReport: Sendable, Equatable, Decodable {
    /// Inputs written, together, for the next tick.
    public let applied: [String]
    /// Inputs the scene refused; each keeps its previous value.
    public let rejected: [TrideKitInputRefusal]
    /// Names in the batch the scene has no input of.
    public let unknown: [String]
    /// Inputs a batch can write that this one did not mention (or, in a strict batch that stopped, did not write).
    public let untouched: [String]

    public init(
        applied: [String] = [], rejected: [TrideKitInputRefusal] = [], unknown: [String] = [],
        untouched: [String] = []
    ) {
        self.applied = applied
        self.rejected = rejected
        self.unknown = unknown
        self.untouched = untouched
    }
}

/// A node under a point of the view (`TrideKitScene.pick(at:)`).
public struct TrideKitHit: Sendable, Equatable, CustomStringConvertible {
    /// The node's id in the scene.
    public let node: String
    /// For a clone or a generated module: which instance was hit.
    public let instanceKey: String?
    public let instanceIndex: Int?
    /// Distance from the camera, in metres.
    public let distance: Float

    public init(node: String, instanceKey: String? = nil, instanceIndex: Int? = nil, distance: Float) {
        self.node = node
        self.instanceKey = instanceKey
        self.instanceIndex = instanceIndex
        self.distance = distance
    }

    init(_ hit: TridePointerHit) {
        self.init(
            node: hit.node,
            instanceKey: hit.instanceKey,
            instanceIndex: hit.instanceIndex.map(Int.init),
            distance: hit.distance
        )
    }

    public var description: String {
        let instance = instanceKey.map { " #\($0)" } ?? instanceIndex.map { " #\($0)" } ?? ""
        return "\(node)\(instance) at \(distance) m"
    }
}

/// Which Tride runtime the app carries — attach it to a bug report.
public struct TrideKitBuildInfo: Sendable, Equatable, CustomStringConvertible {
    /// The release (`0.5.1`).
    public let version: String
    /// The commit the runtime was built from; empty for a build outside the release pipeline.
    public let commit: String
    /// The newest `.trd` format this runtime opens.
    public let formatVersion: String

    public init(version: String, commit: String, formatVersion: String) {
        self.version = version
        self.commit = commit
        self.formatVersion = formatVersion
    }

    init(_ info: TrideBuildInfo) {
        self.init(version: info.version, commit: info.commit, formatVersion: info.formatVersion)
    }

    public var description: String {
        "Tride \(version) (\(commit.isEmpty ? "local build" : commit), format \(formatVersion))"
    }
}

/// Where a host script ended (`TrideKitScene.advance(scriptJSON:)`).
public enum TrideKitAdvance: Sendable, Equatable {
    /// The script ran to its end in `ticks` steps.
    case completed(ticks: Int)
    /// The script stopped at `step` (zero-based) — an `until` ran out of time, an input was refused, or
    /// the script could not be read (step 0).
    case stopped(step: Int, reason: String, ticks: Int)
}

/// A problem the scene ran into that the host should know about (`subscribeDiagnostics`). Each
/// `(kind, subject, message)` is reported once per scene; the log gets it too. `code` names the
/// problem where it has a name every host shares: `material.package-missing`,
/// `material.package-stale`, `material.package-version` (a node material drawn plainly).
public struct TrideKitDiagnostic: Sendable, Equatable, CustomStringConvertible {
    public enum Kind: String, Sendable {
        /// A binding expression failed on a step; `subject` is its pointer.
        case evaluation
        /// Part of the scene could not be prepared (environment, node material package, generated or
        /// module mesh); `subject` is the asset or id.
        case load
        /// A change the view cannot apply (an unknown or out-of-range pointer); `subject` is the pointer
        /// or id.
        case pointer
        /// A call into the runtime failed (`tick`, `pointer`, `setViewport`…); `subject` is the call.
        case runtime
    }

    public let kind: Kind
    public let subject: String
    public let message: String
    public let code: String?

    public init(kind: Kind, subject: String, message: String, code: String? = nil) {
        self.kind = kind
        self.subject = subject
        self.message = message
        self.code = code
    }

    public var description: String { "\(kind.rawValue) \(subject): \(message)" }
}
