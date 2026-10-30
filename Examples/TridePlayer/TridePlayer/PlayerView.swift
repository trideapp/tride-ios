import SwiftUI
import TrideCore
import TrideKit
import UniformTypeIdentifiers

/// The player's one screen: the scene full screen, "Open .trd…", the events the scene reports and
/// a sheet with every binding and trigger the scene declares.
///
/// TrideKit in three steps: `TrideKitScene(trdBytes:)` loads the file, `FilamentMetalView` draws it
/// and hands back its bridge (`attachBridge`), `play()` starts the scene's clock. The host only
/// writes bindings and fires triggers; the scene decides what to show.
struct PlayerView: View {
    @State private var scene: TrideKitScene?
    /// A new view (and Filament engine) for every loaded scene.
    @State private var generation = 0
    @State private var title = ""
    @State private var problem: String?
    @State private var events: [String] = []
    @State private var showsBindings = false
    @State private var showsImporter = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let scene {
                TrideSceneView(
                    scene: scene,
                    // Draw only the frames a tick changed.
                    renderPolicy: .onDemand,
                    onLoaded: { scene.play() },
                    onError: { problem = "The scene did not load: \($0.localizedDescription)" }
                )
                .id(generation)
                .ignoresSafeArea()
            } else {
                Text("Open a .trd exported from Tride Editor")
                    .foregroundStyle(.white.opacity(0.7))
            }
            VStack(spacing: 12) {
                topBar
                Spacer()
                if let problem {
                    Text(problem).font(.footnote).foregroundStyle(.red).padding(.horizontal)
                }
                if !events.isEmpty {
                    Text(events.suffix(3).joined(separator: ", "))
                        .font(.caption.monospaced())
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
            .padding()
        }
        .sheet(isPresented: $showsBindings) {
            if let scene { BindingsPanel(scene: scene) }
        }
        .fileImporter(isPresented: $showsImporter, allowedContentTypes: [.item]) { result in
            if case .success(let url) = result { load(url: url) }
        }
    }

    // MARK: - Controls

    private var topBar: some View {
        HStack {
            Button {
                showsImporter = true
            } label: {
                Label(title.isEmpty ? "Open .trd…" : title, systemImage: "doc")
            }
            Spacer()
            Button("Bindings") { showsBindings = true }
                .disabled(scene == nil)
        }
        .buttonStyle(.bordered)
    }

    // MARK: - Scene

    private func load(url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let loaded = try TrideKitScene(trdBytes: try Data(contentsOf: url))
            scene?.pause()
            loaded.subscribeEvents { event in events.append(event.name) }
            scene = loaded
            generation += 1
            title = url.deletingPathExtension().lastPathComponent
            problem = nil
            events = []
        } catch {
            problem = "Could not open: \(error.localizedDescription)"
        }
    }
}
