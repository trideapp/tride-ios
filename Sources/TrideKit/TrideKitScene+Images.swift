/// The app's pictures in a scene's image inputs: a photo from the library or the network
/// shown in every field the scene binds to the input, in place of the author's placeholder.
///
/// ## Public API
/// - `setImage(_:data:)` — JPEG / PNG / WebP / HEIC bytes, decoded off the main thread to at most
///   2048 px a side, turned by EXIF; `setImage(_:cgImage:)` — a picture already decoded (no EXIF:
///   pass it upright). Both end in a `TrideImageResult`.
/// - Every call throws `TrideError.InvalidImage` for a name that is no image input of the scene —
///   the app's mistake, nothing changes; a picture that cannot be read ends `.failed`. The same
///   rule on Android and the web.
/// - `imageLoading(_:)` — say a picture is on its way before its bytes are here (a download, the
///   photo picker): the scene sees `'loading'`; a photo already shown stays until the next one.
/// - `clearImage(_:)` — back to no picture: the placeholder. `imageInput(_:)` — the status, revision
///   and size the input shows now.
/// - Status changes reach `subscribeEvents` as `image.<input>` with `{status, revision, width,
///   height, reason?}`.
///
/// ## Non-obvious constraints
/// - The latest call per input wins: a decode a later `setImage` / `imageLoading` / `clearImage` of
///   the same input overtook ends `.superseded` and never reaches the scene.
/// - A picture shows from the step after `.loaded` (the core lays it into the fields on a tick).
/// - Pixels stay in the store while a slot shows them or the input shows that revision (a
///   replacement loading keeps the previous photo on screen); then they go.
/// - Failures go to the runtime (`imageFailed`, the placeholder comes back) and to
///   `subscribeDiagnostics` once per `(input, reason)`.

import CoreGraphics
import Foundation

#if canImport(TrideCore)
import TrideCore
#endif

#if canImport(TrideKitFilamentBridge)

@available(iOS 17.0, macCatalyst 17.0, *)
extension TrideKitScene {

    /// Show `data` (JPEG / PNG / WebP / HEIC) in image input `name`. Marks the input `'loading'`
    /// at once — a photo already shown stays meanwhile — decodes off the main thread to at most
    /// 2048 px a side, turned upright by its EXIF orientation, and hands it to the scene. Bytes
    /// that do not decode end `.failed`; throws `TrideError.InvalidImage`, before anything
    /// changes, when `name` is no image input of the scene.
    public func setImage(_ name: String, data: Data) async throws -> TrideImageResult {
        try await deliverImage(name) { try HostImageDecoder.decode(data) }
    }

    /// Show an already decoded picture in image input `name` — the same path as `setImage(_:data:)`,
    /// scaled down to 2048 px a side when larger. A `CGImage` has no orientation: pass it upright.
    public func setImage(_ name: String, cgImage: CGImage) async throws -> TrideImageResult {
        try await deliverImage(name) { try HostImageDecoder.picture(from: cgImage) }
    }

    /// A picture for image input `name` is on its way (a download, the photo picker): the scene sees
    /// `'loading'`, a photo already shown stays. Supersedes a decode still running for `name`.
    /// Throws `TrideError.InvalidImage`, changing nothing, when `name` is no image input.
    public func imageLoading(_ name: String) throws {
        TrideKitThreadRule.requireMain()
        try trideScene.imageLoading(name: name)
        _ = imageRequests.begin(name)
    }

    /// Back to no picture in image input `name`: its fields show the placeholder from the next step.
    /// Supersedes a decode still running for `name`. Throws `TrideError.InvalidImage`, changing
    /// nothing, when `name` is no image input.
    public func clearImage(_ name: String) throws {
        TrideKitThreadRule.requireMain()
        try trideScene.clearImage(name: name)
        _ = imageRequests.begin(name)
        releaseUnshownPictures()
    }

    /// What image input `name` shows now; `nil` for a name that is no image input.
    public func imageInput(_ name: String) -> TrideImageInput? {
        TrideKitThreadRule.requireMain()
        return trideScene.imageInput(name: name)
    }

    // MARK: - Internal

    /// One request: `'loading'` (the name checked by the runtime), the decode off the main thread,
    /// then — unless a later request of `name` came first — the picture to the scene and the store.
    private func deliverImage(
        _ name: String, decode: @escaping @Sendable () throws -> HostPicture
    ) async throws -> TrideImageResult {
        try trideScene.imageLoading(name: name)
        let token = imageRequests.begin(name)
        let decoded = await Task.detached(priority: .userInitiated) { Result(catching: decode) }.value
        guard imageRequests.isLatest(token, for: name) else { return .superseded }
        do {
            let picture = try decoded.get()
            let revision = try trideScene.setImage(
                name: name, width: UInt32(picture.width), height: UInt32(picture.height))
            hostImages.put(picture, for: HostImageKey(name: name, revision: revision))
            releaseUnshownPictures()
            return .loaded
        } catch {
            let why = reason(of: error)
            do {
                try trideScene.imageFailed(name: name, reason: why)
            } catch {
                imageCallFailed("imageFailed", name, error)
            }
            note(TrideKitDiagnostic(kind: .load, subject: "image \(name)", message: "\(why) — the placeholder stays"))
            deliverDiagnostics()
            releaseUnshownPictures()
            return .failed(reason: why)
        }
    }

    /// Let go of every picture no slot shows and whose input no longer shows that revision.
    func releaseUnshownPictures() {
        guard !hostImages.isEmpty else { return }
        var keep = dispatcher.boundHostPictures
        for name in hostImages.names {
            guard let input = trideScene.imageInput(name: name), input.width > 0,
                  input.status == "loaded" || input.status == "loading"
            else { continue }
            keep.insert(HostImageKey(name: name, revision: input.revision))
        }
        hostImages.retain(only: keep)
    }

    private func imageCallFailed(_ call: String, _ name: String, _ error: Error) {
        note(TrideKitDiagnostic(kind: .runtime, subject: "\(call) \(name)", message: reason(of: error)))
        deliverDiagnostics()
    }

    private func reason(of error: Error) -> String {
        switch error {
        case let failure as HostImageDecoder.Failure: return failure.reason
        case TrideError.InvalidImage(let message): return message
        default: return String(describing: error)
        }
    }
}

#endif
