/// One finger of a touch view as the scene's pointer — the rule `PointerMTKView` follows.
///
/// ## Public API
/// None: `FingerPhase` and `TrideKitScene.finger(_:at:)` are internal to TrideKit.
///
/// ## Non-obvious constraints
/// - A finger has no hover (W3C Pointer Events). Its lift is `up` — the tap or the drag ends, and a
///   tap still clicks — then `cancel`: the scene hears the pointer leave, its `pointerExit` listeners
///   run and the scene's own rule decides what a follow does next (hold, or return).
/// - A drag behaves as before: a `cancel` after the release throws nothing.
/// - The same rule as TrideKit for Android and as a browser's touch (`pointerleave` after `pointerup`).

import CoreGraphics

#if canImport(TrideCore)
import TrideCore
#endif

/// What the tracked finger did, in UIKit's terms.
enum FingerPhase {
    case began, moved, ended, cancelled

    /// The pointer phases the scene hears for it, in order.
    var pointerPhases: [TridePointerPhase] {
        switch self {
        case .began: return [.down]
        case .moved: return [.move]
        case .ended: return [.up, .cancel]
        case .cancelled: return [.cancel]
        }
    }
}

#if canImport(TrideKitFilamentBridge)

@available(iOS 17.0, macCatalyst 17.0, *)
extension TrideKitScene {
    /// The scene hears the finger's `phase` at `point` (view points from the top-left).
    func finger(_ phase: FingerPhase, at point: CGPoint) {
        for pointerPhase in phase.pointerPhases {
            pointer(pointerPhase, at: point)
        }
    }
}

#endif
