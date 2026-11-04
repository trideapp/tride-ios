/// The app's pictures for a scene's image inputs: decoded off the main thread, kept by
/// `(name, revision)` while something shows them.
///
/// ## Public API
/// - `TrideImageResult` — how `TrideKitScene.setImage` ended: `.loaded`, `.failed(reason:)`, `.superseded`.
/// - `TrideHostPictureProbe` (`@_spi(TrideTesting)`) — the decoder's pixels for Tride's test hosts.
///
/// ## Internal
/// - `HostPicture` — one picture as straight-alpha sRGB RGBA8; `HostImageKey` — `(name, revision)`.
/// - `HostImageDecoder` — ImageIO, at most 2048 px a side, turned by its EXIF orientation.
/// - `HostImageStore` — the one store of the app's pictures (R-OneLookupPath): photo slots read it,
///   later the environment and L-system pens.
/// - `HostImageRequests` — the latest request per input wins; an older decode is dropped.
///
/// ## Non-obvious constraints
/// - Straight alpha in sRGB, as glTF textures and the web host (`premultiplyAlpha: 'none'`); vImage
///   converts, never a `CGBitmapContext` (premultiplied only).
/// - A picture within 2048 px that needs no turn is read whole (`CreateImageAtIndex`), its alpha
///   exact: ImageIO on iOS hands a PNG back premultiplied (on macOS not), so such pixels are read
///   again by the bridge's one picture decoder (`FilamentBridge.decodedPicturePixels`, stb) and laid
///   in the picture's own colour space — the sRGB conversion stays ImageIO's. WebP / HEIC
///   (stb reads neither) keep ImageIO's premultiplied pixels.
/// - Otherwise a thumbnail — `kCGImageSourceCreateThumbnailFromImageAlways`, not `…IfAbsent` (that
///   is the file's embedded preview, a JPEG's ~160 px EXIF thumbnail, `CGImageSource.h`); ImageIO
///   draws it premultiplied: there a pixel of low alpha loses some colour precision.

import Accelerate
import CoreGraphics
import Foundation
import ImageIO

#if canImport(TrideKitFilamentBridge)
import TrideKitFilamentBridge
#endif

/// How `TrideKitScene.setImage` ended.
public enum TrideImageResult: Sendable, Equatable {
    /// The scene has the picture: its image input is `"loaded"` and its slots show it from the next
    /// step.
    case loaded
    /// The picture could not be used (unreadable bytes, an input that is no image input): the input
    /// is `"failed"` and its slots show the placeholder again.
    case failed(reason: String)
    /// A later `setImage` / `imageLoading` / `clearImage` of the same input came first: this picture
    /// was dropped and the input is what that later call made it.
    case superseded
}

/// One of the app's pictures as the slots take it.
struct HostPicture: Sendable, Equatable {
    let width: Int
    let height: Int
    /// Straight-alpha sRGB RGBA8, rows top to bottom, `width * 4` bytes each.
    let pixels: Data
}

/// Which picture of an image input: `revision` is what `TrideScene.setImage` returned for it.
struct HostImageKey: Hashable, Sendable, CustomStringConvertible {
    let name: String
    let revision: UInt64

    var description: String { "\(name)@\(revision)" }
}

/// Bytes or a `CGImage` → `HostPicture` within the core's 2048 px a side.
enum HostImageDecoder {
    /// The longest side the core accepts (`TrideError.InvalidImage` past it) — also the size every
    /// GPU Filament runs on guarantees.
    static let maxPixelSize = 2048

    struct Failure: Error, Equatable, CustomStringConvertible {
        let reason: String
        var description: String { reason }
    }

    /// A JPEG / PNG / WebP / HEIC (whatever ImageIO reads) at most 2048 px a side, turned upright.
    static func decode(_ data: Data) throws -> HostPicture {
        guard !data.isEmpty else { throw Failure(reason: "no picture bytes") }
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, sourceOptions),
              CGImageSourceGetCount(source) > 0
        else { throw Failure(reason: "not a picture this device reads (JPEG / PNG / WebP / HEIC)") }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let width = (properties?[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
        let height = (properties?[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        if width > 0, height > 0, max(width, height) <= maxPixelSize, orientation == 1,
           let whole = CGImageSourceCreateImageAtIndex(
            source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        {
            return try straightRGBA(storedAlpha(of: whole, in: data) ?? whole)
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw Failure(reason: "the picture could not be decoded")
        }
        return try straightRGBA(image)
    }

    /// `image`'s pixels as the file stores them when ImageIO premultiplied them: the bridge's picture
    /// decoder (stb) reads `data` again, laid in `image`'s colour space (sRGB for a grey or indexed
    /// one: stb hands those back as RGB). `nil` = keep `image` — its alpha is already straight, or the
    /// decoder does not read the format, or it read another size.
    static func storedAlpha(of image: CGImage, in data: Data) -> CGImage? {
        switch image.alphaInfo {
        case .premultipliedLast, .premultipliedFirst: break
        default: return nil
        }
        #if canImport(TrideKitFilamentBridge)
        var size = CGSize.zero
        guard let pixels = FilamentBridge.decodedPicturePixels(data, size: &size),
              Int(size.width) == image.width, Int(size.height) == image.height,
              let space = image.colorSpace.flatMap({ $0.model == .rgb ? $0 : nil })
                ?? CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: pixels as CFData)
        else { return nil }
        return CGImage(
            width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: image.width * 4, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        #else
        return nil
        #endif
    }

    /// `image` as it is when it fits in 2048 px a side, else drawn down to fit, proportions kept.
    static func picture(from image: CGImage) throws -> HostPicture {
        try straightRGBA(fitted(image))
    }

    private static func fitted(_ image: CGImage) throws -> CGImage {
        let width = image.width, height = image.height
        guard width > 0, height > 0 else { throw Failure(reason: "an empty picture") }
        let longest = max(width, height)
        guard longest > maxPixelSize else { return image }
        let scale = Double(maxPixelSize) / Double(longest)
        let w = max(1, min(maxPixelSize, Int((Double(width) * scale).rounded())))
        let h = max(1, min(maxPixelSize, Int((Double(height) * scale).rounded())))
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw Failure(reason: "no memory to scale the picture") }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let scaled = context.makeImage() else { throw Failure(reason: "the picture could not be scaled") }
        return scaled
    }

    private static func straightRGBA(_ image: CGImage) throws -> HostPicture {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let format = vImage_CGImageFormat(
                bitsPerComponent: 8, bitsPerPixel: 32, colorSpace: space,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                renderingIntent: .defaultIntent)
        else { throw Failure(reason: "no sRGB colour space") }
        var buffer: vImage_Buffer
        do {
            buffer = try vImage_Buffer(cgImage: image, format: format)
        } catch {
            throw Failure(reason: "the picture could not be converted to RGBA (\(error))")
        }
        defer { buffer.free() }
        let width = Int(buffer.width), height = Int(buffer.height)
        guard width > 0, height > 0, let rows = buffer.data else { throw Failure(reason: "an empty picture") }
        let row = width * 4
        var pixels = Data(count: row * height)
        pixels.withUnsafeMutableBytes { (out: UnsafeMutableRawBufferPointer) in
            guard let base = out.baseAddress else { return }
            for y in 0..<height {
                memcpy(base + y * row, rows + y * buffer.rowBytes, row)
            }
        }
        return HostPicture(width: width, height: height, pixels: pixels)
    }
}

/// What the decoder makes of `data` — the pixels a photo slot uploads — for Tride's own test hosts
/// (the simulator's ImageIO differs from the Mac's).
@_spi(TrideTesting) public enum TrideHostPictureProbe {
    /// Straight-alpha sRGB RGBA8, rows top to bottom, or the decoder's failure.
    public static func decode(_ data: Data) throws -> (width: Int, height: Int, pixels: Data) {
        let picture = try HostImageDecoder.decode(data)
        return (picture.width, picture.height, picture.pixels)
    }
}

/// The one store of the app's pictures by `(name, revision)` (R-OneLookupPath). Its owner lets a
/// picture go once no slot shows it and its input no longer shows that revision.
final class HostImageStore {
    private var pictures: [HostImageKey: HostPicture] = [:]

    var isEmpty: Bool { pictures.isEmpty }
    var keys: Set<HostImageKey> { Set(pictures.keys) }
    /// The inputs with a picture held.
    var names: Set<String> { Set(pictures.keys.map(\.name)) }
    /// Pixel bytes held.
    var byteCount: Int { pictures.values.reduce(0) { $0 + $1.pixels.count } }

    func put(_ picture: HostPicture, for key: HostImageKey) {
        pictures[key] = picture
    }

    func picture(for key: HostImageKey) -> HostPicture? {
        pictures[key]
    }

    /// Let go of every picture not in `keep`; returns how many went.
    @discardableResult
    func retain(only keep: Set<HostImageKey>) -> Int {
        let before = pictures.count
        pictures = pictures.filter { keep.contains($0.key) }
        return before - pictures.count
    }
}

/// The latest request per image input: each `begin` supersedes the input's earlier ones.
struct HostImageRequests {
    private var latest: [String: UInt64] = [:]
    private var issued: UInt64 = 0

    /// A new request of `name`; the token to check when its decode is done.
    mutating func begin(_ name: String) -> UInt64 {
        issued += 1
        latest[name] = issued
        return issued
    }

    func isLatest(_ token: UInt64, for name: String) -> Bool {
        latest[name] == token
    }
}
