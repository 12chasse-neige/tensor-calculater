import AppKit
import Foundation
import SwiftUI
import SwiftMath
import XCTest
@testable import TensorCalculator

@MainActor
final class NativeIntegrationTests: XCTestCase {
    func testPackagedWorkerComputesSphereThroughNativeModel() async throws {
        guard let workerPath = ProcessInfo.processInfo.environment["TENSOR_INTEGRATION_WORKER_PATH"] else {
            throw XCTSkip("Set TENSOR_INTEGRATION_WORKER_PATH to validate a real packaged calculation worker.")
        }
        let previous = ProcessInfo.processInfo.environment["TENSOR_WORKER_PATH"]
        setenv("TENSOR_WORKER_PATH", workerPath, 1)
        defer {
            if let previous { setenv("TENSOR_WORKER_PATH", previous, 1) }
            else { unsetenv("TENSOR_WORKER_PATH") }
        }
        let inputs = try XCTUnwrap(MetricPreset.all.first(where: { $0.id == "sphere" })).inputs
        let model = CalculationModel()
        model.calculate(inputs)
        try await waitUntil { !model.isRunning }
        XCTAssertNil(model.errorMessage)
        let result = try XCTUnwrap(model.result)
        XCTAssertEqual(result.coordinates, ["theta", "phi"])
        XCTAssertEqual(result.scalars.first(where: { $0.key == "ricci_scalar" })?.expression, "-2/L**2")
        XCTAssertEqual(result.scalars.first(where: { $0.key == "kretschmann" })?.expression, "4/L**4")
        XCTAssertEqual(Set(result.tensors.map(\.key)), ["inverse_metric", "christoffel", "riemann", "ricci"])
        XCTAssertEqual(model.resultInputs, inputs)

        // This renders views we own into memory. It never opens or captures a
        // desktop window and is available even when UI automation is unavailable.
        if let snapshotDirectory = ProcessInfo.processInfo.environment["TENSOR_SNAPSHOT_DIR"] {
            let directory = URL(fileURLWithPath: snapshotDirectory, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            var document = CalculationDocument()
            try await snapshot(document: document, to: directory.appendingPathComponent("flat-editor-dark.png"), colorScheme: .dark)
            try await snapshotContent(CalculationView(document: .constant(document)), to: directory.appendingPathComponent("flat-editor-small.png"),
                                      size: CGSize(width: 900, height: 680), expectedLabels: 5)
            document.inputs.riemannSign = -1
            document.inputs.metricSign = -1
            try await snapshot(document: document, to: directory.appendingPathComponent("opposite-conventions.png"))
            document.inputs = inputs
            try await snapshot(document: document, to: directory.appendingPathComponent("metric-editor.png"))
            try await snapshot(document: document, to: directory.appendingPathComponent("metric-editor-dark.png"), colorScheme: .dark)
            document.result = result
            document.resultInputs = inputs
            try await snapshot(document: document, to: directory.appendingPathComponent("sphere-results.png"))
            try await snapshot(document: document, to: directory.appendingPathComponent("sphere-results-dark.png"), colorScheme: .dark)
            for key in ["christoffel", "riemann"] {
                let tensor = try XCTUnwrap(result.tensors.first(where: { $0.key == key }))
                try await snapshotContent(TensorComponentsView(tensor: tensor, coordinates: result.coordinates, search: .constant("")),
                                          to: directory.appendingPathComponent(key + "-indices.png"), colorScheme: .dark)
            }
            let longExpression = String(repeating: "r^2 + ", count: 30_000)
            document.result = CalculationResult(coordinates: result.coordinates, elapsed_seconds: result.elapsed_seconds,
                convention: result.convention, tensors: [],
                scalars: [ScalarResult(key: "ricci_scalar", name: "Ricci scalar", symbol: "R", expression: longExpression, latex: longExpression)], warnings: [])
            try await snapshot(document: document, to: directory.appendingPathComponent("long-expression.png"))
        }
    }

    func testLatexMetricComputesWithBothSignControls() async throws {
        guard let worker = ProcessInfo.processInfo.environment["TENSOR_INTEGRATION_WORKER_PATH"] else { throw XCTSkip("Set integration worker path") }
        let previous = ProcessInfo.processInfo.environment["TENSOR_WORKER_PATH"]
        setenv("TENSOR_WORKER_PATH", worker, 1)
        defer {
            if let previous { setenv("TENSOR_WORKER_PATH", previous, 1) }
            else { unsetenv("TENSOR_WORKER_PATH") }
        }
        var inputs = try XCTUnwrap(MetricPreset.all.first(where: { $0.id == "sphere" })).inputs
        inputs.metric = try MetricNotation.latex(inputs.metric)
        inputs.metricSign = -1
        inputs.riemannSign = -1
        let model = CalculationModel()
        model.calculate(inputs)
        try await waitUntil { !model.isRunning }
        XCTAssertNil(model.errorMessage)
        XCTAssertEqual(model.result?.scalars.first(where: { $0.key == "ricci_scalar" })?.expression, "-2/L**2")
        XCTAssertEqual(model.result?.scalars.first(where: { $0.key == "kretschmann" })?.expression, "4/L**4")
    }

    func testLatexInputAndSignRequests() throws {
        let latex = #"\begin{pmatrix}-\frac{1}{r} & 0 \\ 0 & r^{2}\sin^{2}{\theta}\end{pmatrix}"#
        let rows = try MetricNotation.rows(latex)
        XCTAssertEqual(rows.count, 2)
        XCTAssertTrue(rows[0][0].contains("/"))
        XCTAssertTrue(rows[1][1].contains("sin( theta )^(2)"))
        XCTAssertEqual(try MetricNotation.expression(#"\sin\left(\theta\right)"#), "sin( theta )")
        XCTAssertTrue(try MetricNotation.latex("[r^(2*(n+1))]").contains("r^{2 (n+1)}"))
        XCTAssertTrue(try MetricNotation.latex("[r^0.5]").contains("r^{0.5}"))
        XCTAssertThrowsError(try MetricNotation.expression(#"\unknown{r}"#))
        XCTAssertThrowsError(try MetricNotation.expression(#"\frac{1}"#))
        XCTAssertThrowsError(try MetricNotation.rows("[1, 0], garbage [0, 1]"))
        var inputs = CalculationInputs(coordinates: "r, theta", metric: latex)
        XCTAssertNil(inputs.validationMessage)
        inputs.metricSign = -1
        inputs.riemannSign = -1
        let request = CalculationRequest(id: "sign-test", inputs: inputs)
        XCTAssertEqual(request.riemann_sign, -1)
        XCTAssertTrue(request.metric.hasPrefix("[-("))
        let old = #"{"coordinates":"x","scalars":"","functions":"","metric":"[1]","outputs":["ricci"]}"#
        let decoded = try JSONDecoder().decode(CalculationInputs.self, from: Data(old.utf8))
        XCTAssertEqual(CalculationRequest(id: "old", inputs: decoded).riemann_sign, 1)
        XCTAssertNil(decoded.metricSign)
        var document = CalculationDocument()
        document.inputs = inputs
        XCTAssertEqual(try CalculationDocument(data: document.encodedData()).inputs, inputs)
    }

    func testLargeFormulaPolicyBoundsAutomaticParsingAndUnicodePreview() {
        XCTAssertFalse(FormulaSizePolicy.isLong(latex: "\\frac{2}{L^2}", expression: "2/L^2"))
        let huge = String(repeating: "r^2 + ", count: 30_000)
        XCTAssertTrue(FormulaSizePolicy.isLong(latex: huge, expression: "0"))
        XCTAssertTrue(FormulaSizePolicy.isLong(latex: "0", expression: huge))
        let preview = FormulaSizePolicy.preview(huge)
        XCTAssertLessThanOrEqual(preview.count, FormulaSizePolicy.maximumPreviewCharacters + 2)
        XCTAssertTrue(preview.hasSuffix("\n…"))
        XCTAssertEqual(FormulaSizePolicy.preview("r^2"), "r^2")
        let unicode = String(repeating: "θ", count: 2_000)
        XCTAssertEqual(FormulaSizePolicy.preview(unicode), String(repeating: "θ", count: 1_200) + "\n…")
    }

    func testNativeTypesetterRendersMatricesDerivativesAndLongRationalExpressions() throws {
        let formulas = [
            "\\begin{pmatrix}-1 & 0 \\\\ 0 & r^{2}\\end{pmatrix}",
            "\\frac{d^{2}}{dt^{2}} a\\left(t\\right)",
            "\\frac{\\partial^{2}}{\\partial t\\partial r}\\Phi\\left(t,r\\right)",
            "\\Gamma^{r}_{\\theta\\theta} = -r\\left(1-\\frac{2 M}{r}\\right)",
            "\\frac{48 M^{2}\\left(r^{6}-15 a^{2}r^{4}\\cos^{2}{\\theta}+15 a^{4}r^{2}\\cos^{4}{\\theta}-a^{6}\\cos^{6}{\\theta}\\right)}{\\left(r^{2}+a^{2}\\cos^{2}{\\theta}\\right)^{6}}"
        ]
        for formula in formulas {
            let label = MTMathUILabel()
            label.latex = formula
            XCTAssertNil(label.error, "Unsupported formula: \(formula)")
            XCTAssertGreaterThan(label.fittingSize.width, 0)
            XCTAssertGreaterThan(label.fittingSize.height, 0)
        }
    }

    func testPresetSelectionKeepsHighlightAndMetricInSync() async throws {
        let store = TestDocumentStore()
        store.document.inputs.metricSign = -1
        store.document.inputs.riemannSign = -1
        let view = NSHostingView(rootView: CalculationView(document: Binding(
            get: { store.document }, set: { store.document = $0 }
        )))
        let bounds = NSRect(x: 0, y: 0, width: 1180, height: 820)
        let window = NSWindow(contentRect: bounds, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.frame = bounds
        defer { window.close() }
        try await Task.sleep(nanoseconds: 150_000_000)
        view.layoutSubtreeIfNeeded()
        let table = try XCTUnwrap(tableView(in: view))
        // Two section headers, the Metric row, then the six preset rows.
        XCTAssertEqual(table.numberOfRows, MetricPreset.all.count + 3)
        for id in ["rn", "schwarzschild", "kerr", "flrw", "flat", "sphere", "rn"] {
            let index = try XCTUnwrap(MetricPreset.all.firstIndex(where: { $0.id == id }))
            let row = index + 3
            table.selectRowIndexes(IndexSet(integer: row), byExtendingSelection: false)
            try await Task.sleep(nanoseconds: 70_000_000)
            XCTAssertEqual(table.selectedRow, row, "Only the clicked example should be highlighted: " + id)
            XCTAssertTrue(MetricPreset.all[index].matchesDefinition(store.document.inputs), "Selected example and displayed metric must agree: " + id)
            XCTAssertEqual(store.document.inputs.metricSign, -1)
            XCTAssertEqual(store.document.inputs.riemannSign, -1)
        }
    }

    private final class TestDocumentStore: ObservableObject {
        @Published var document = CalculationDocument()
    }

    private func tableView(in view: NSView) -> NSTableView? {
        if let table = view as? NSTableView { return table }
        return view.subviews.compactMap { tableView(in: $0) }.first
    }

    func testIndexSlotsAndConventionLabelsUseValidLatex() throws {
        let formulas = [TensorNotation.christoffel, TensorNotation.riemann, TensorNotation.ricciContraction, #"g_{ij}"#]
            + RiemannConvention.allCases.flatMap { [$0.buttonLatex, $0.definitionLatex] }
        for formula in formulas {
            let label = MTMathUILabel()
            label.latex = formula
            XCTAssertNil(label.error, formula)
            XCTAssertGreaterThan(label.fittingSize.height, 0)
        }
        XCTAssertEqual(TensorNotation.indexed("T", labels: ["a", "b", "c"], variance: "lul"), #"T_{a}{}^{b}{}_{c}"#)
        let tensor = TensorResult(key: "riemann", name: "Riemann", symbol: "R", variance: "ulll", rank: 4, shape: [4,4,4,4], components: [])
        let component = TensorComponent(indices: [0,1,2,3], expression: "1", latex: "1", zero_status: "nonzero")
        XCTAssertEqual(tensor.label(for: component, coordinates: ["mu", "nu", "rho", "sigma"]), TensorNotation.riemann)
        XCTAssertEqual(RiemannConvention(resultConvention: "R^rho_{sigma mu nu} = d_mu Gamma"), .derivativeFirstFirst)
    }

    func testDocumentKeepsTheInputsThatProducedItsSavedResult() throws {
        var document = CalculationDocument()
        document.resultInputs = document.inputs
        document.result = sampleResult
        document.inputs.scalars = "M"
        let reopened = try CalculationDocument(data: document.encodedData())
        XCTAssertEqual(reopened.inputs.scalars, "M")
        XCTAssertEqual(reopened.resultInputs?.scalars, "")
        XCTAssertEqual(reopened.result, sampleResult)
        XCTAssertThrowsError(try CalculationDocument(data: Data("{\"version\":2,\"inputs\":{}}".utf8)))
    }

    func testFragmentedProtocolAndFullStderrPipeCompleteWithoutDeadlock() async throws {
        try await withWorker(body: """
        sys.stderr.write("diagnostic " * 10000)
        sys.stderr.flush()
        stale = dict(version=1,id="stale",type="error",message="ignore this")
        print(json.dumps(stale), flush=True)
        progress = json.dumps(dict(version=1,id=request["id"],type="progress",stage="ricci",message="Computing Ricci")) + "\\n"
        for part in [progress[:10], progress[10:]]:
            sys.stdout.write(part)
            sys.stdout.flush()
        result = \(sampleResultJSON)
        assert request["version"] == 1
        assert request["outputs"]
        sys.stdout.write(json.dumps(dict(version=1,id=request["id"],type="result",result=result)))
        sys.stdout.flush()
        """) { _, _ in
            let model = CalculationModel()
            model.calculate(CalculationInputs())
            try await waitUntil { !model.isRunning }
            XCTAssertNil(model.errorMessage)
            XCTAssertEqual(model.result?.scalars.first?.expression, "7")
            XCTAssertFalse(model.diagnostic.isEmpty)
            XCTAssertLessThanOrEqual(model.diagnostic.utf8.count, 64 * 1024)
        }
    }

    func testExitWithoutResultFailsEvenWhenOldResultHasIdenticalInputs() async throws {
        try await withWorker(body: "pass") { _, _ in
            let model = CalculationModel()
            var document = CalculationDocument()
            document.result = sampleResult
            document.resultInputs = document.inputs
            model.restore(document)
            model.calculate(document.inputs)
            try await waitUntil { !model.isRunning }
            XCTAssertEqual(model.errorMessage, "The calculation engine finished without a result.")
        }
    }

    func testMalformedResponseFailsClearly() async throws {
        try await withWorker(body: "print('not JSON', flush=True)") { _, _ in
            let model = CalculationModel()
            model.calculate(CalculationInputs())
            try await waitUntil { !model.isRunning }
            XCTAssertTrue(model.errorMessage?.contains("unreadable response") == true)
        }
    }

    func testCancellationKillsAnUncooperativeWorkerAndRejectsLateMessages() async throws {
        try await withWorker(body: """
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with open(sys.argv[0] + '.pid','w') as file:
            file.write(str(os.getpid()))
        print(json.dumps(dict(version=1,id=request["id"],type="progress",message="Waiting")),flush=True)
        time.sleep(0.5)
        print(json.dumps(dict(version=1,id=request["id"],type="error",message="late message")),flush=True)
        time.sleep(30)
        """) { url, _ in
            let model = CalculationModel()
            model.calculate(CalculationInputs())
            try await waitUntil { model.status == "Waiting" }
            let pid = Int32(try String(contentsOfFile: url.path + ".pid", encoding: .utf8))
            XCTAssertNotNil(pid)
            model.cancel()
            XCTAssertFalse(model.isRunning)
            XCTAssertEqual(model.status, "Cancelled")
            try await waitUntil { kill(pid!, 0) == -1 }
            XCTAssertNil(model.errorMessage)
            XCTAssertEqual(model.status, "Cancelled")
        }
    }

    func testLabelsPreserveIndexPositionsAndGreekCoordinates() {
        let tensor = TensorResult(key: "christoffel", name: "Christoffel", symbol: "\\Gamma", variance: "ull", rank: 3, shape: [3,3,3], components: [])
        let component = TensorComponent(indices: [1,2,2], expression: "-r", latex: "-r", zero_status: "unknown")
        XCTAssertEqual(tensor.label(for: component, coordinates: ["t", "r", "theta"]), "\\Gamma^{r}{}_{\\theta\\,\\theta}")
    }

    private var sampleResult: CalculationResult {
        try! JSONDecoder().decode(CalculationResult.self, from: Data(sampleResultJSON.utf8))
    }
    private var sampleResultJSON: String {
        """
        {"coordinates":["t","r","theta","phi"],"elapsed_seconds":0.1,"convention":"R^a_bcd = d_c Gamma^a_db - d_d Gamma^a_cb + ...","tensors":[],"scalars":[{"key":"ricci_scalar","name":"Ricci scalar","symbol":"R","expression":"7","latex":"7"}],"warnings":[]}
        """
    }

    private func waitUntil(_ predicate: () -> Bool) async throws {
        for _ in 0..<800 {
            if predicate() { return }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTFail("Timed out waiting for calculation state")
        throw NSError(domain: "Tests", code: 1)
    }

    private func snapshot(document: CalculationDocument, to path: URL, colorScheme: ColorScheme = .light) async throws {
        let expected = document.result?.scalars.filter {
            !FormulaSizePolicy.isLong(latex: $0.latex, expression: $0.expression)
        }.count ?? 0
        try await snapshotContent(CalculationView(document: .constant(document)), to: path, colorScheme: colorScheme, expectedLabels: expected)
    }

    private func snapshotContent<Content: View>(_ content: Content, to path: URL, colorScheme: ColorScheme = .light, size: CGSize = CGSize(width: 1180, height: 820), expectedLabels: Int = 0) async throws {
        // AppKit sidebar cells resolve vibrancy against the application's
        // appearance when they are created, even in an offscreen window.
        let application = NSApplication.shared
        let previousAppearance = application.appearance
        let appearance = NSAppearance(named: colorScheme == .dark ? .darkAqua : .aqua)
        application.appearance = appearance
        defer { application.appearance = previousAppearance }
        let bounds = NSRect(origin: .zero, size: size)
        let view = NSHostingView(rootView: content
            .background(Color(nsColor: .windowBackgroundColor))
            .environment(\.colorScheme, colorScheme)
            .frame(width: bounds.width, height: bounds.height))
        let window = NSWindow(contentRect: bounds, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = appearance
        window.backgroundColor = .windowBackgroundColor
        window.contentView = view
        view.appearance = appearance
        view.frame = bounds
        defer { window.close() }
        try await Task.sleep(nanoseconds: 150_000_000)
        view.layoutSubtreeIfNeeded()
        let labels = mathLabels(in: view)
        XCTAssertGreaterThanOrEqual(labels.count, expectedLabels)
        for label in labels {
            XCTAssertEqual(label.mathLabel.textColor, colorScheme == .dark ? NSColor.white : NSColor.black,
                           "Native formula glyph colors must follow the SwiftUI appearance.")
            XCTAssertGreaterThanOrEqual(label.frame.width + 0.5, label.fittingSize.width, "A native formula must receive its full fitting width.")
            XCTAssertGreaterThanOrEqual(label.frame.height + 0.5, label.fittingSize.height, "A native formula must receive its full fitting height.")
        }
        let bitmap = try XCTUnwrap(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let data = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        XCTAssertGreaterThan(data.count, 10_000, "The view snapshot should contain rendered content.")
        try data.write(to: path)
    }

    private func mathLabels(in view: NSView) -> [NativeMathCanvas] {
        let own = (view as? NativeMathCanvas).map { [$0] } ?? []
        return own + view.subviews.flatMap { mathLabels(in: $0) }
    }

    private func withWorker(body: String, run: (URL, URL) async throws -> Void) async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tensor-native-tests-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("worker.py")
        let source = "#!/usr/bin/env python3\nimport sys, json, time, os\nrequest = json.loads(sys.stdin.readline())\n" + body + "\n"
        try source.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        let previous = ProcessInfo.processInfo.environment["TENSOR_WORKER_PATH"]
        setenv("TENSOR_WORKER_PATH", url.path, 1)
        defer {
            if let previous { setenv("TENSOR_WORKER_PATH", previous, 1) }
            else { unsetenv("TENSOR_WORKER_PATH") }
        }
        try await run(url, directory)
    }
}
