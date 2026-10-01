import SwiftUI

struct CalculationView: View {
    @Environment(\.colorScheme) private var colorScheme
    @Binding var document: CalculationDocument
    @StateObject private var model = CalculationModel()
    @State private var selection: String? = "input"
    @State private var search = ""
    @State private var restored = false

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 215, ideal: 250, max: 320)
        } detail: {
            detail
                .navigationTitle(CalculationNavigation.showsInputs(selection) ? "Metric" : "Results")
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                if model.isRunning {
                    Button { model.cancel() } label: { Label("Cancel", systemImage: "stop.fill") }
                        .keyboardShortcut(".", modifiers: .command)
                }
                Button {
                    model.calculate(document.inputs)
                } label: { Label("Calculate", systemImage: "play.fill") }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(model.isRunning || document.inputs.validationMessage != nil)
                .help(document.inputs.validationMessage ?? "Calculate the selected tensors (⌘↩)")
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) { statusBar }
        .alert("Calculation failed", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.dismissError() } }
        )) {
            Button("OK") { model.dismissError() }
        } message: { Text(model.errorMessage ?? "") }
        .onAppear {
            guard !restored else { return }
            restored = true
            model.restore(document)
            if document.result != nil { selection = "overview" }
            else if let preset = MetricPreset.all.first(where: { $0.matchesDefinition(document.inputs) }) {
                selection = CalculationNavigation.presetKey(preset.id)
            }
        }
        .onDisappear { model.cancel() }
        .onChange(of: model.result) { _, result in
            if document.result != result { document.result = result }
            if result != nil { selection = "overview" }
        }
        .onChange(of: model.resultInputs) { _, inputs in
            if document.resultInputs != inputs { document.resultInputs = inputs }
        }
    }

    private var sidebar: some View {
        List(selection: Binding(get: { selection }, set: { selected in
            guard !model.isRunning || CalculationNavigation.preset(for: selected) == nil else { return }
            if let preset = CalculationNavigation.preset(for: selected) {
                document.inputs = preset.applying(to: document.inputs)
            }
            selection = selected
        })) {
            Section {
                navigationRow("Metric", systemImage: "square.grid.3x3", key: "input")
                if model.result != nil {
                    navigationRow("Overview", systemImage: "chart.bar.doc.horizontal", key: "overview")
                }
            } header: { Text("Calculation").foregroundStyle(sidebarText.opacity(0.65)) }
            if let result = model.result {
                Section("Tensors") {
                    ForEach(result.tensors) { tensor in
                        navigationRow(tensor.name, systemImage: "square.stack.3d.up", key: "tensor:" + tensor.key)
                    }
                    ForEach(result.scalars) { scalar in
                        navigationRow(scalar.name, systemImage: "function", key: "scalar:" + scalar.key)
                    }
                }
            }
            Section {
                ForEach(MetricPreset.all) { preset in
                    VStack(alignment: .leading, spacing: 3) {
                        Text(preset.name).foregroundStyle(presetText(for: preset))
                        Text(preset.detail).font(.caption).foregroundStyle(presetText(for: preset).opacity(0.75))
                    }
                    .padding(.vertical, 3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
                    .tag(CalculationNavigation.presetKey(preset.id))
                    .accessibilityIdentifier("preset-" + preset.id)
                    .disabled(model.isRunning)
                }
            } header: { Text("Presets").foregroundStyle(sidebarText.opacity(0.65)) }
        }
        .listStyle(.sidebar)
    }

    private var sidebarText: Color { colorScheme == .dark ? .white : .black }

    private func presetText(for preset: MetricPreset) -> Color {
        selection == CalculationNavigation.presetKey(preset.id) ? .white : sidebarText
    }

    private func navigationRow(_ title: String, systemImage: String, key: String) -> some View {
        Label(title, systemImage: systemImage)
            .foregroundStyle(selection == key ? Color.white : sidebarText)
            .tag(key)
    }

    @ViewBuilder
    private var detail: some View {
        if CalculationNavigation.showsInputs(selection) || model.result == nil {
            InputView(inputs: $document.inputs)
        } else if let result = model.result {
            VStack(spacing: 0) {
                if model.resultInputs != document.inputs {
                    Label("These results belong to the previous metric. Calculate again to update them.", systemImage: "clock.arrow.circlepath")
                        .font(.callout)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                        .background(.yellow.opacity(0.10))
                }
                ResultBrowser(result: result, selection: selection ?? "overview", search: $search) { key in
                    selection = key
                }
            }
        }
    }

    private var statusBar: some View {
        HStack(spacing: 9) {
            if model.isRunning {
                ProgressView().controlSize(.small)
            } else {
                Image(systemName: model.errorMessage == nil ? "circle.fill" : "exclamationmark.circle")
                    .font(.system(size: 7))
                    .foregroundStyle(model.errorMessage == nil ? Color.secondary : Color.orange)
            }
            Text(model.status).font(.caption).lineLimit(1)
            Spacer()
            Text("\(document.inputs.coordinateNames.count) coordinates")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 9)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }
}

private struct InputView: View {
    @Binding var inputs: CalculationInputs

    private var selectedConvention: RiemannConvention {
        RiemannConvention(rawValue: inputs.riemannSign ?? 1) ?? .derivativeLastFirst
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Define your spacetime").font(.title.weight(.semibold))
                    Text("Enter the covariant metric in coordinate order, then choose the quantities to calculate.")
                        .foregroundStyle(.secondary)
                }
                GroupBox("Coordinates and symbols") {
                    Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 12) {
                        GridRow {
                            Text("Coordinates")
                            TextField("t, r, theta, phi", text: $inputs.coordinates)
                        }
                        GridRow {
                            Text("Constants")
                            TextField("M, a, k", text: $inputs.scalars)
                        }
                        GridRow {
                            Text("Functions")
                            TextField("a(t), Phi(t,r)", text: $inputs.functions)
                        }
                    }
                    .textFieldStyle(.roundedBorder)
                    .padding(8)
                }
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Metric").font(.headline)
                        InlineMathFormula(latex: #"g_{ij}"#, fontSize: 18)
                            .accessibilityElement(children: .ignore)
                            .accessibilityLabel("g subscript i j")
                        Spacer()
                        Button("Use LaTeX") {
                            if let latex = try? MetricNotation.latex(inputs.metric) { inputs.metric = latex }
                        }
                        .disabled((try? MetricNotation.rows(inputs.metric)) == nil)
                        .help("Convert the metric to a LaTeX matrix")
                    }
                    HStack(alignment: .top, spacing: 16) {
                        MetricEditor(text: $inputs.metric)
                            .frame(height: CGFloat(min(180, max(95, inputs.coordinateNames.count * 25 + 24))))
                            .frame(maxWidth: .infinity)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay { RoundedRectangle(cornerRadius: 8).stroke(.separator, lineWidth: 1) }
                        if let preview = try? MetricNotation.latex(inputs.metric) {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Preview · " + inputs.coordinateNames.joined(separator: ", "))
                                    .font(.caption).foregroundStyle(.secondary)
                                MathFormula(latex: "g_{ij} = " + (inputs.metricSign == -1 ? "-" : "") + preview,
                                            expression: "Covariant metric", fontSize: 19, showExpression: false)
                            }
                            .padding(12)
                            .frame(width: 280, alignment: .leading)
                            .frame(minHeight: 70, alignment: .topLeading)
                            .background(.background, in: RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    Text("Bracketed rows or LaTeX matrix / pmatrix / bmatrix. LaTeX supports \\frac{a}{b}, \\sqrt{a}, \\sin{theta}^{2}, and Greek symbols.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                GroupBox("Sign conventions") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 12) {
                            Text("Metric").frame(width: 70, alignment: .leading)
                            ConventionButton(text: inputs.coordinateNames.count == 4 ? "(− + + +)" : "g",
                                             accessibilityLabel: "Metric: minus plus plus plus; use entered matrix",
                                             selected: inputs.metricSign != -1) { inputs.metricSign = 1 }
                            ConventionButton(text: inputs.coordinateNames.count == 4 ? "(+ − − −)" : "−g",
                                             accessibilityLabel: "Metric: plus minus minus minus; reverse all signs",
                                             selected: inputs.metricSign == -1) { inputs.metricSign = -1 }
                        }
                        Text(inputs.coordinateNames.count == 4
                             ? "Enter the matrix in the (− + + +) convention. Choosing (+ − − −) reverses all entries."
                             : "Use the entered matrix g or reverse all its signs with −g.")
                            .font(.caption).foregroundStyle(.secondary)
                        HStack(spacing: 12) {
                            Text("Riemann").frame(width: 70, alignment: .leading)
                            ForEach(RiemannConvention.allCases) { convention in
                                ConventionButton(latex: convention.buttonLatex,
                                                 accessibilityLabel: convention.accessibilityLabel,
                                                 selected: (inputs.riemannSign ?? 1) == convention.rawValue) {
                                    inputs.riemannSign = convention.rawValue
                                }
                            }
                        }
                        MathFormula(latex: selectedConvention.definitionLatex,
                                    expression: selectedConvention.accessibilityLabel, fontSize: 17, showExpression: false)
                        MathFormula(latex: TensorNotation.ricciContraction,
                                    expression: "Ricci contraction", fontSize: 16, showExpression: false)
                    }.padding(8)
                }
                GroupBox("Calculate") {
                    LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)], alignment: .leading, spacing: 12) {
                        ForEach(CalculationOutput.allCases) { output in
                            Toggle(output.title, isOn: Binding(
                                get: { inputs.outputs.contains(output.rawValue) },
                                set: { selected in
                                    if selected {
                                        if !inputs.outputs.contains(output.rawValue) { inputs.outputs.append(output.rawValue) }
                                    } else { inputs.outputs.removeAll { $0 == output.rawValue } }
                                }
                            ))
                        }
                    }
                    .toggleStyle(.checkbox)
                    .padding(8)
                }
                if let message = inputs.validationMessage {
                    Label(message, systemImage: "info.circle").foregroundStyle(.secondary).font(.callout)
                }
            }
            .padding(28)
            .frame(maxWidth: 1000, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }
}

private struct ResultBrowser: View {
    let result: CalculationResult
    let selection: String
    @Binding var search: String
    let navigate: (String) -> Void

    var body: some View {
        if let tensor = result.tensors.first(where: { "tensor:" + $0.key == selection }) {
            TensorComponentsView(tensor: tensor, coordinates: result.coordinates, search: $search)
        } else if let scalar = result.scalars.first(where: { "scalar:" + $0.key == selection }) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text(scalar.name).font(.largeTitle.weight(.semibold))
                    ScalarCard(scalar: scalar)
                    convention
                }
                .padding(28)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    HStack(alignment: .firstTextBaseline) {
                        Text("Calculation results").font(.largeTitle.weight(.semibold))
                        Spacer()
                        Text(String(format: "%.2f seconds", result.elapsed_seconds))
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    Text("Coordinates: " + result.coordinates.joined(separator: ", "))
                        .foregroundStyle(.secondary)
                    ForEach(result.scalars) { scalar in ScalarCard(scalar: scalar) }
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 235), alignment: .leading)], spacing: 14) {
                        ForEach(result.tensors) { tensor in
                            Button { navigate("tensor:" + tensor.key) } label: {
                                VStack(alignment: .leading, spacing: 9) {
                                    HStack {
                                        Image(systemName: "square.stack.3d.up").foregroundStyle(.tint)
                                        Spacer()
                                        Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
                                    }
                                    Text(tensor.name).font(.headline)
                                    Text("\(tensor.components.count) stored components")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(16)
                                .background(.background, in: RoundedRectangle(cornerRadius: 10))
                                .overlay { RoundedRectangle(cornerRadius: 10).stroke(.separator, lineWidth: 1) }
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    if !result.warnings.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(Array(result.warnings.enumerated()), id: \.offset) { _, warning in
                                Label(warning, systemImage: "info.circle").font(.callout)
                            }
                        }
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                    }
                    convention
                }
                .padding(28)
            }
        }
    }

    private var convention: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Curvature convention").font(.headline)
            if let convention = RiemannConvention(resultConvention: result.convention) {
                MathFormula(latex: convention.definitionLatex,
                            expression: result.convention, fontSize: 17, showExpression: false)
                MathFormula(latex: TensorNotation.ricciContraction,
                            expression: "Ricci contraction", fontSize: 16, showExpression: false)
            } else {
                Text(result.convention).font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary).textSelection(.enabled)
            }
            Text("Components whose zero value is established symbolically are omitted.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

private struct ScalarCard: View {
    let scalar: ScalarResult
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(scalar.name).font(.headline)
                Spacer()
                CopyMenu(expression: scalar.expression, latex: scalar.latex)
            }
            MathFormula(latex: scalar.symbol + " = " + scalar.latex, expression: scalar.symbol + " = " + scalar.expression, fontSize: 26)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: 10))
        .overlay { RoundedRectangle(cornerRadius: 10).stroke(.separator, lineWidth: 1) }
    }
}

struct TensorComponentsView: View {
    let tensor: TensorResult
    let coordinates: [String]
    @Binding var search: String
    private var filtered: [TensorComponent] {
        guard !search.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return tensor.components }
        return tensor.components.filter { component in
            (tensor.label(for: component, coordinates: coordinates) + " " + component.id + " " + component.expression)
                .localizedCaseInsensitiveContains(search)
        }
    }

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 8) {
                    Text(tensor.name).font(.largeTitle.weight(.semibold))
                    Text("Rank \(tensor.rank) · Shape " + tensor.shape.map(String.init).joined(separator: " × ") + " · \(filtered.count) of \(tensor.components.count) components")
                        .font(.callout).foregroundStyle(.secondary)
                    if tensor.key == "christoffel" || tensor.key == "riemann" {
                        MathFormula(latex: tensor.key == "christoffel" ? TensorNotation.christoffel : TensorNotation.riemann,
                                    expression: "One upper index, followed by " + (tensor.key == "christoffel" ? "two" : "three") + " lower indices in component order.",
                                    fontSize: 23)
                    }
                    Text("Index order: " + coordinates.enumerated().map { "\($0.offset) = \($0.element)" }.joined(separator: ", "))
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.bottom, 6)
                if filtered.isEmpty {
                    ContentUnavailableView(
                        search.isEmpty ? "All components vanish" : "No matching components",
                        systemImage: search.isEmpty ? "checkmark.circle" : "magnifyingglass",
                        description: Text(search.isEmpty ? "No nonzero components are stored for this tensor." : "Search by coordinates, numeric indices, or expression text.")
                    )
                } else {
                    ForEach(filtered) { component in
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("Indices (\(component.id))").font(.caption).foregroundStyle(.secondary)
                                if ["unknown", "undetermined"].contains(component.zero_status) {
                                    Text("Zero unresolved").font(.caption2).foregroundStyle(.secondary)
                                        .help("This expression is retained because its zero value has not been proved symbolically.")
                                }
                                Spacer()
                                CopyMenu(expression: component.expression, latex: tensor.label(for: component, coordinates: coordinates) + " = " + component.latex)
                            }
                            MathFormula(
                                latex: tensor.label(for: component, coordinates: coordinates) + " = " + component.latex,
                                expression: component.expression
                            )
                        }
                        .padding(16)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.background, in: RoundedRectangle(cornerRadius: 9))
                        .overlay { RoundedRectangle(cornerRadius: 9).stroke(.separator, lineWidth: 1) }
                    }
                }
            }
            .padding(28)
        }
        .searchable(text: $search, prompt: "Coordinates, indices, or expression")
    }
}

private struct CopyMenu: View {
    let expression: String
    let latex: String
    var body: some View {
        Menu {
            Button("Copy expression") { copyToClipboard(expression) }
            Button("Copy LaTeX") { copyToClipboard(latex) }
        } label: { Label("Copy", systemImage: "doc.on.doc") }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Copy the expression or LaTeX")
    }
}
