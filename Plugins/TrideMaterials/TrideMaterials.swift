/// TrideMaterials — a build-tool plugin that precompiles the node-material packages of the app's
/// `.trd` scenes into Metal libraries on the developer's Mac, so an iPhone's first run loads them
/// instead of compiling Metal source.
///
/// Public API: apply the plugin to the app target that bundles the `.trd` files (Xcode: Build Phases →
/// Run Build Tool Plug-ins → TrideMaterials; SwiftPM: `plugins: [.plugin(name: "TrideMaterials",
/// package: "TrideKit")]`).
///
/// Non-obvious constraints:
///   - Output = side files `tride-metal-<sha256>.filamat` in the bundle, keyed by the sha256 of the
///     package bytes as the `.trd` holds them; the `.trd` itself is not touched. TrideKit looks the key
///     up on a device; the simulator keeps the package's Metal source (it cannot load iphoneos libraries).
///   - A prebuild command runs on every build: packages whose key is already built are skipped, keys no
///     scene uses any more are removed.
///   - The compile hook is the one Tride builds its own materials with (fixed source name, iOS
///     minimum = TrideKit's): the same package compiles to the same bytes everywhere.
import Foundation
import PackagePlugin

@main
struct TrideMaterials: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
        guard let module = target.sourceModule else { return [] }
        let scenes = module.sourceFiles.filter { $0.path.extension == "trd" }.map(\.path)
        return try Self.commands(
            scenes: scenes, work: context.pluginWorkDirectory, matedit: context.tool(named: "matedit").path)
    }

    /// TrideKit's minimum iOS (`Package.swift` platforms): a library built for a newer iOS than the app
    /// runs on fails to load its functions.
    static let minimumIOS = "17.0"
    static let minimumMacOS = "14.0"

    static func commands(scenes: [Path], work: Path, matedit: Path) -> [Command] {
        guard !scenes.isEmpty else { return [] }
        let output = work.appending("materials")
        return [
            .prebuildCommand(
                displayName: "TrideMaterials: Metal libraries for \(scenes.count) scene(s)",
                executable: Path("/bin/sh"),
                arguments: ["-c", script, "tride-materials", output.string, work.appending("work").string,
                            matedit.string, minimumIOS, minimumMacOS] + scenes.map(\.string),
                outputFilesDirectory: output)
        ]
    }

    /// `$1` output, `$2` scratch, `$3` matedit, `$4` / `$5` minimum iOS / macOS, then the scenes.
    static let script = #"""
        set -eu
        out="$1"; work="$2"; matedit="$3"; min_ios="$4"; min_macos="$5"; shift 5
        mkdir -p "$out" "$work"
        export TMPDIR="$work"
        hook="$work/metal-compile.sh"
        cat > "$hook" <<'HOOK'
        #!/bin/sh
        set -eu
        in="$1"; out="$2"; platform="$4"; min_ios="$5"; min_macos="$6"
        tmp=$(mktemp -d)
        trap 'rm -rf "$tmp"' EXIT
        cp "$in" "$tmp/shader.metal"
        if [ "$platform" = "mobile" ]; then
          sdk=iphoneos; min="-mios-version-min=$min_ios"
        else
          sdk=macosx; min="-mmacosx-version-min=$min_macos"
        fi
        xcrun -sdk "$sdk" metal -x metal -c "$min" -o "$tmp/shader.air" "$tmp/shader.metal"
        xcrun -sdk "$sdk" metallib -o "$out" "$tmp/shader.air"
        HOOK
        chmod +x "$hook"
        one="$work/one.sh"
        cat > "$one" <<'ONE'
        #!/bin/sh
        set -eu
        out="$1"; work="$2"; matedit="$3"; hook="$4"; min_ios="$5"; min_macos="$6"; trd="$7"; entry="$8"
        pkg=$(mktemp "$work/pkg.XXXXXX")
        trap 'rm -f "$pkg" "$pkg.metal"' EXIT
        unzip -p "$trd" "$entry" > "$pkg"
        name="tride-metal-$(shasum -a 256 "$pkg" | cut -d' ' -f1).filamat"
        echo "$name" >> "$work/keep"
        [ -f "$out/$name" ] && exit 0
        "$matedit" -t metal -p -i "$pkg" -o "$pkg.metal" external-compile -- "$hook" "$min_ios" "$min_macos" >/dev/null
        mv "$pkg.metal" "$out/$name"
        ONE
        : > "$work/keep"
        for trd; do
          unzip -Z1 "$trd" | grep '\.filamat$' | while IFS= read -r entry; do
            printf '%s\0%s\0' "$trd" "$entry"
          done
        done | xargs -0 -n 2 -P 8 sh "$one" "$out" "$work" "$matedit" "$hook" "$min_ios" "$min_macos"
        for f in "$out"/tride-metal-*.filamat; do
          [ -e "$f" ] || continue
          grep -qx "$(basename "$f")" "$work/keep" || rm -f "$f"
        done
        echo "TrideMaterials: $(sort -u "$work/keep" | wc -l | tr -d ' ') Metal-library package(s)"
        """#
}

#if canImport(XcodeProjectPlugin)
import XcodeProjectPlugin

extension TrideMaterials: XcodeBuildToolPlugin {
    func createBuildCommands(context: XcodePluginContext, target: XcodeTarget) throws -> [Command] {
        let scenes = target.inputFiles.filter { $0.path.extension == "trd" }.map(\.path)
        return try Self.commands(
            scenes: scenes, work: context.pluginWorkDirectory, matedit: context.tool(named: "matedit").path)
    }
}
#endif
