// swift-tools-version:5.10
/// TrideKit — plays Tride scenes (`.trd`) in an iOS app: the Tride runtime (`TrideCore`, a binary
/// xcframework), the Filament renderer (a binary xcframework) and a SwiftUI view over them.
///
/// ## Public API
/// - `TrideKit` — `TrideKitScene` (load a `.trd`, bindings, triggers, events, play / pause, which
///   runtime build — `buildInfo`) and `FilamentMetalView` (the SwiftUI view that draws it).
/// - `TrideCore` — the runtime's own types (`TrideValue`, `TrideBindingDescriptor`, …).
/// - `TrideMaterials` — a build-tool plugin for the app target that bundles `.trd` scenes: their
///   materials ship as precompiled Metal libraries, so an iPhone's first run does not compile them.
///
/// ## Non-obvious constraints
/// - Generated from the Tride repository (`SNAPSHOT.md`): do not edit it here — the next release
///   replaces it.
/// - iOS 17+, arm64 only (device and Apple-silicon simulator). On macOS (arm64 only) there is no
///   Filament: `TrideKit` compiles to a re-export of `TrideCore`.
/// - No unsafe flags: the ObjC++ bridge to Filament ships compiled (`TrideKitFilamentBridge`), so the
///   package can be added by URL. Its materials ride in TrideKit's resources.

import PackageDescription

let package = Package(
    name: "TrideKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "TrideKit", targets: ["TrideKit"]),
        .library(name: "TrideCore", targets: ["TrideCore"]),
        .plugin(name: "TrideMaterials", targets: ["TrideMaterials"]),
    ],
    targets: [
        // The Tride runtime (Rust), built for iOS, the iOS simulator and macOS.
        .binaryTarget(
            name: "trideFFI",
            url: "https://api.github.com/repos/trideapp/tride-ios/releases/assets/709730454.zip",
            checksum: "612e9234431e8561cc13f4c21d8fec3f2432fd152b0e604a630e6eba3b209c2e"
        ),
        // Google Filament (Apache-2.0), built from Google's source of the pinned release, its
        // libraries merged into one.
        .binaryTarget(
            name: "Filament",
            url: "https://api.github.com/repos/trideapp/tride-ios/releases/assets/709730640.zip",
            checksum: "c2bc5eb90c9ba191cd261a11c4c25d10d1e23e57d06de1464173f8a0ec9f28e0"
        ),
        // The thin ObjC++ bridge to Filament, compiled with `-fno-rtti` like Filament itself.
        .binaryTarget(
            name: "TrideKitFilamentBridge",
            url: "https://api.github.com/repos/trideapp/tride-ios/releases/assets/709730820.zip",
            checksum: "adf96bc2bbbfab8e48a4bcfdf1bfbe2266a81e5e985dac2eb51ad30a96670410"
        ),
        // Filament's `matedit` (Google's official release of the same version, Apache-2.0) for the
        // TrideMaterials plugin.
        .binaryTarget(
            name: "matedit",
            url: "https://api.github.com/repos/trideapp/tride-ios/releases/assets/709730924.zip",
            checksum: "a024f0903a2136c36b034f14f61d8f8ac46ffcade3916a03562ced561751e8ad"
        ),
        // Precompiles the materials of the app's `.trd` scenes into Metal libraries at the app's build.
        .plugin(name: "TrideMaterials", capability: .buildTool(), dependencies: ["matedit"]),
        // Swift bindings of the runtime (uniffi output).
        .target(
            name: "TrideCore",
            dependencies: ["trideFFI"],
            path: "Sources/TrideCore"
        ),
        .target(
            name: "TrideKit",
            dependencies: [
                "TrideCore",
                .target(name: "TrideKitFilamentBridge", condition: .when(platforms: [.iOS])),
                .target(name: "Filament", condition: .when(platforms: [.iOS])),
            ],
            path: "Sources/TrideKit",
            // `.process` lays the bridge's materials at the bundle's root, where it looks them up.
            resources: [.copy("Resources/ibl"), .process("Resources/Materials")],
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("ImageIO"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("Foundation"),
                .linkedFramework("UIKit", .when(platforms: [.iOS])),
            ]
        ),
    ]
)
