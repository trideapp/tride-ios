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
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.18.0/TrideCoreFFI.xcframework.zip",
            checksum: "a43cda5ccc1cca4eddf73be22c1ea141ebda741f7f79532a54baabd615435e06"
        ),
        // Google Filament (Apache-2.0), built from Google's source of the pinned release, its
        // libraries merged into one.
        .binaryTarget(
            name: "Filament",
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.18.0/Filament.xcframework.zip",
            checksum: "01f477b5e48cd5ad95af5a26b17124001331c5f5d89ab1052f85b9d666edc08e"
        ),
        // The thin ObjC++ bridge to Filament, compiled with `-fno-rtti` like Filament itself.
        .binaryTarget(
            name: "TrideKitFilamentBridge",
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.18.0/TrideKitFilamentBridge.xcframework.zip",
            checksum: "74a2a47950cd3e762d30867aca20c3639167674062aaefb06a4ac69faea8c888"
        ),
        // Filament's `matedit` (Google's official release of the same version, Apache-2.0) for the
        // TrideMaterials plugin.
        .binaryTarget(
            name: "matedit",
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.18.0/matedit.artifactbundle.zip",
            checksum: "279433844b6a243d31228daeed67d8fff82aa65500cb1fa19ed49b9881ad47b9"
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
