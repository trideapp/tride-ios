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
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.15.0/TrideCoreFFI.xcframework.zip",
            checksum: "30111c7d0b9c7003ad65abe36e33b4e0b86fc2fba22985c492b2cc1442514d06"
        ),
        // Google Filament (Apache-2.0), built from Google's source of the pinned release, its
        // libraries merged into one.
        .binaryTarget(
            name: "Filament",
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.15.0/Filament.xcframework.zip",
            checksum: "fe2a373fad2dd862d1909cd01dd9c820f9394982fff60d210bd788175f16db0b"
        ),
        // The thin ObjC++ bridge to Filament, compiled with `-fno-rtti` like Filament itself.
        .binaryTarget(
            name: "TrideKitFilamentBridge",
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.15.0/TrideKitFilamentBridge.xcframework.zip",
            checksum: "3e4d076ae14d2b6fc9f89540c1aa5a7af2eb67e52df654ef667974a0ff336f7d"
        ),
        // Filament's `matedit` (Google's official release of the same version, Apache-2.0) for the
        // TrideMaterials plugin.
        .binaryTarget(
            name: "matedit",
            url: "https://github.com/trideapp/tride-ios/releases/download/v0.15.0/matedit.artifactbundle.zip",
            checksum: "51e2c87082c200bd0e1434b4d512bb7176b73e2648f1a1ed79021255ec777671"
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
