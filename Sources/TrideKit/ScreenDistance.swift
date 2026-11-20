/// How far a scene view stands from the screen — what it draws and what GPU
/// memory it keeps, decided from its frame alone, as `IntersectionObserver`'s `rootMargin` on the
/// web or `UICollectionView` prefetching do: the bridge knows nothing of feeds or pages.
///
/// ## API (module-internal; `FilamentMetalView` takes it on every display frame)
/// - `ScreenDistance.of(frame:screen:)` — `.onScreen` when the view's frame (window coordinates)
///   shows any pixel of `screen`; `.near` within one screen of its edge (a feed's neighbouring page,
///   exactly one screen aside); `.far` beyond.
/// - What each means for the view: on screen — draws; near — draws nothing, keeps its swap chain and
///   renderer (coming back costs nothing); far — gives them back, the layer keeps its last frame. On a
///   memory warning near views give them back too and far ones shrink their layer (the last frame
///   goes: more than a screen away, they come back in time).
///
/// ## Non-obvious constraints
/// - An empty frame or screen (before the first layout) counts as on screen: the view behaves as it
///   always did until it has a place.
/// - A frame touching the screen's edge does not show on it: the next page of a paged feed is near.

import CoreGraphics

enum ScreenDistance: Equatable, Sendable {
    case onScreen
    case near
    case far

    static func of(frame: CGRect, screen: CGRect) -> ScreenDistance {
        if frame.isNull || frame.isEmpty || screen.isNull || screen.isEmpty { return .onScreen }
        if frame.intersects(screen) { return .onScreen }
        let reach = screen.insetBy(dx: -screen.width, dy: -screen.height)
        return frame.intersects(reach) ? .near : .far
    }
}
