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
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.19.1/TrideCoreFFI.xcframework.zip",
            checksum: "1b18078e1e6fc8f4449ac0b75ed641d90685ea7214c8d98200e8f4f8b3a1742b"
        ),
        // Google Filament (Apache-2.0), built from Google's source of the pinned release, its
        // libraries merged into one.
        .binaryTarget(
            name: "Filament",
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.19.1/Filament.xcframework.zip",
            checksum: "0da39857fbd842d53b91d7bacfc8601097dd8c2349a225b9c546f8aeca852aec"
        ),
        // The thin ObjC++ bridge to Filament, compiled with `-fno-rtti` like Filament itself.
        .binaryTarget(
            name: "TrideKitFilamentBridge",
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.19.1/TrideKitFilamentBridge.xcframework.zip",
            checksum: "1353a85f8500f79f39a157a9f6d1f3c87afb62c94c2593dfa47d4dfa90fae1b2"
        ),
        // Filament's `matedit` (Google's official release of the same version, Apache-2.0) for the
        // TrideMaterials plugin.
        .binaryTarget(
            name: "matedit",
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.19.1/matedit.artifactbundle.zip",
            checksum: "8407d458c81973c9ab15e8ccafcf1373f60ca9595b6e6fc606263a8dd311ca88"
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
