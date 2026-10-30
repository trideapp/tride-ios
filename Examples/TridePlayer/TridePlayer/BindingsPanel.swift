import SwiftUI
import TrideCore
import TrideKit

/// Every binding and trigger the loaded scene declares (`availableBindings()`), each with a control
/// that writes it: a trigger is a button, a bool a toggle, an enum a picker, a number a field.
/// Vector and colour bindings are listed but not edited here.
struct BindingsPanel: View {
    let scene: TrideKitScene

    @Environment(\.dismiss) private var dismiss
    @State private var numbers: [String: Double] = [:]
    @State private var flags: [String: Bool] = [:]
    @State private var choices: [String: String] = [:]
    @State private var problem: String?

    var body: some View {
        NavigationStack {
            List {
                if let problem {
                    Text(problem).foregroundStyle(.red)
                }
                let bindings = scene.availableBindings()
                Section("Triggers") {
                    ForEach(bindings.filter { $0.kind == .trigger }, id: \.name) { binding in
                        Button(binding.name) { run { try scene.trigger(binding.name) } }
                    }
                }
                Section("Values") {
                    ForEach(bindings.filter { $0.kind != .trigger }, id: \.name) { binding in
                        row(binding)
                    }
                }
            }
            .navigationTitle("Scene bindings")
            .toolbar { Button("Done") { dismiss() } }
        }
        .presentationDetents([.medium, .large])
    }

    @ViewBuilder
    private func row(_ binding: TrideKitBinding) -> some View {
        if binding.computed {
            // The scene computes it: show the value, never write it.
            HStack {
                Text(binding.name)
                Spacer()
                Text((try? scene.getBinding(binding.name)).map(Self.text) ?? "—").foregroundStyle(.secondary)
            }
        } else {
            editor(binding)
        }
    }

    @ViewBuilder
    private func editor(_ binding: TrideKitBinding) -> some View {
        switch binding.kind {
        case .bool:
            Toggle(binding.name, isOn: Binding(
                get: { flags[binding.name] ?? false },
                set: { value in
                    flags[binding.name] = value
                    run { try scene.setBinding(binding.name, .bool(value: value)) }
                }
            ))
        case .enum:
            Picker(binding.name, selection: Binding(
                get: { choices[binding.name] ?? binding.options.first ?? "" },
                set: { value in
                    choices[binding.name] = value
                    run { try scene.setBinding(binding.name, .string(value: value)) }
                }
            )) {
                ForEach(binding.options, id: \.self) { Text($0).tag($0) }
            }
        case .number:
            HStack {
                Text(binding.name)
                Spacer()
                TextField("number", value: Binding(
                    get: { numbers[binding.name] ?? 0 },
                    set: { value in
                        numbers[binding.name] = value
                        run { try scene.setBinding(binding.name, .number(value: value)) }
                    }
                ), format: .number)
                .keyboardType(.decimalPad)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 120)
            }
        default:
            HStack {
                Text(binding.name)
                Spacer()
                Text(binding.kind.rawValue).foregroundStyle(.secondary)
            }
        }
    }

    private static func text(_ value: TrideValue) -> String {
        switch value {
        case .number(let n): n.formatted(.number.precision(.fractionLength(0...3)))
        case .bool(let b): b ? "on" : "off"
        case .string(let s): s
        case .vec3(let v): v.map { $0.formatted(.number.precision(.fractionLength(0...3))) }.joined(separator: ", ")
        }
    }

    private func run(_ write: () throws -> Void) {
        do {
            try write()
            problem = nil
        } catch {
            problem = error.localizedDescription
        }
    }
}
