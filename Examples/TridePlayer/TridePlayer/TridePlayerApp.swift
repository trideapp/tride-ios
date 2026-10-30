import SwiftUI

/// TridePlayer — plays a Tride scene (`.trd`) with TrideKit: any `.trd` exported from Tride Editor,
/// with a control for every binding and trigger it declares.
@main
struct TridePlayerApp: App {
    var body: some Scene {
        WindowGroup {
            PlayerView()
                .preferredColorScheme(.dark)
        }
    }
}
