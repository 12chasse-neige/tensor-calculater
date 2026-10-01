import Foundation
import SwiftUI
import UniformTypeIdentifiers

enum CalculationOutput: String, Codable, CaseIterable, Identifiable {
    case inverseMetric = "inverse_metric"
    case christoffel, riemann, ricci
    case ricciScalar = "ricci_scalar"
    case kretschmann
    var id: String { rawValue }
    var title: String {
        switch self {
        case .inverseMetric: "Inverse metric"
        case .christoffel: "Christoffel symbols"
        case .riemann: "Riemann tensor"
        case .ricci: "Ricci tensor"
        case .ricciScalar: "Ricci scalar"
        case .kretschmann: "Kretschmann scalar"
        }
    }
}

struct CalculationInputs: Codable, Equatable {
    var coordinates = "t, r, theta, phi"
    var scalars = ""
    var functions = ""
    var metric = "[-1, 0, 0, 0],\n[0, 1, 0, 0],\n[0, 0, r^2, 0],\n[0, 0, 0, r^2*sin(theta)^2]"
    var outputs = CalculationOutput.allCases.map(\.rawValue)

    var metricSign: Int? = nil
    var riemannSign: Int? = nil // nil preserves version-1 documents

    var coordinateNames: [String] {
        coordinates.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }
    var validationMessage: String? {
        let names = coordinateNames
        guard !names.isEmpty else { return "Enter at least one coordinate." }
        guard !coordinates.split(separator: ",", omittingEmptySubsequences: false).contains(where: {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }) else { return "Separate coordinates with commas; remove empty entries." }
        guard names.allSatisfy({ $0.range(of: "^[A-Za-z_][A-Za-z0-9_]*$", options: .regularExpression) != nil }) else {
            return "Coordinate names must start with a letter or underscore and contain only letters, digits, or underscores."
        }
        guard Set(names).count == names.count else { return "Coordinate names must be unique." }
        guard !metric.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return "Enter a metric matrix." }
        do {
            let rows = try MetricNotation.rows(metric)
            guard rows.count == names.count, rows.allSatisfy({ $0.count == names.count }) else { return "Metric dimensions must match the coordinates." }
        } catch { return error.localizedDescription }
        guard !outputs.isEmpty else { return "Choose at least one output." }
        guard outputs.allSatisfy({ CalculationOutput(rawValue: $0) != nil }) else { return "This document contains an unsupported calculation output." }
        return nil
    }
}

struct CalculationRequest: Encodable {
    let version = 1
    let id: String
    let coordinates: String
    let scalars: String
    let functions: String
    let metric: String
    let outputs: [String]
    let riemann_sign: Int
    init(id: String, inputs: CalculationInputs) {
        self.id = id
        coordinates = inputs.coordinates
        scalars = inputs.scalars
        functions = inputs.functions
        metric = (try? (inputs.metricSign == -1 ? MetricNotation.negated(inputs.metric) : MetricNotation.plain(inputs.metric))) ?? inputs.metric
        riemann_sign = inputs.riemannSign ?? 1
        outputs = inputs.outputs
    }
}

struct TensorComponent: Codable, Equatable, Identifiable {
    let indices: [Int]
    let expression: String
    let latex: String
    let zero_status: String
    var id: String { indices.map(String.init).joined(separator: ",") }
}

struct TensorResult: Codable, Equatable, Identifiable {
    let key: String
    let name: String
    let symbol: String
    let variance: String
    let rank: Int
    let shape: [Int]
    let components: [TensorComponent]
    var id: String { key }

    func label(for component: TensorComponent, coordinates: [String]) -> String {
        let labels = component.indices.map { index in
            let name = coordinates.indices.contains(index) ? coordinates[index] : String(index)
            let greek = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
            if greek.contains(name) { return "\\" + name }
            if name.count == 1 { return name }
            return "\\mathrm{" + name.replacingOccurrences(of: "_", with: "\\_") + "}"
        }
        return TensorNotation.indexed(symbol, labels: labels, variance: variance)
    }
}

struct ScalarResult: Codable, Equatable, Identifiable {
    let key: String
    let name: String
    let symbol: String
    let expression: String
    let latex: String
    var id: String { key }
}

struct CalculationResult: Codable, Equatable {
    let coordinates: [String]
    let elapsed_seconds: Double
    let convention: String
    let tensors: [TensorResult]
    let scalars: [ScalarResult]
    let warnings: [String]
}

struct WorkerEvent: Decodable {
    let version: Int
    let id: String
    let type: String
    let stage: String?
    let message: String?
    let result: CalculationResult?
}

extension UTType {
    static let tensorCalculation = UTType(exportedAs: "org.tensorcalculator.calculation", conformingTo: .json)
}

struct CalculationDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.tensorCalculation, .json] }
    static var writableContentTypes: [UTType] { [.tensorCalculation] }
    var version = 1
    var inputs = CalculationInputs()
    var result: CalculationResult?
    // These inputs identify the exact metric that produced a saved result.
    var resultInputs: CalculationInputs?

    init() {}
    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else { throw CocoaError(.fileReadCorruptFile) }
        try self.init(data: data)
    }
    init(data: Data) throws {
        let stored = try JSONDecoder().decode(StoredCalculation.self, from: data)
        guard stored.version == 1 else {
            throw NSError(domain: "TensorCalculator", code: 1, userInfo: [NSLocalizedDescriptionKey: "This calculation document uses an unsupported format version."])
        }
        version = stored.version
        inputs = stored.inputs
        result = stored.result
        resultInputs = stored.resultInputs
    }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: try encodedData())
    }
    func encodedData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(StoredCalculation(version: version, inputs: inputs, result: result, resultInputs: resultInputs))
    }
    private struct StoredCalculation: Codable {
        let version: Int
        let inputs: CalculationInputs
        let result: CalculationResult?
        let resultInputs: CalculationInputs?
    }
}

struct MetricPreset: Identifiable {
    let id: String
    let name: String
    let detail: String
    let inputs: CalculationInputs
    func matchesDefinition(_ current: CalculationInputs) -> Bool {
        inputs.coordinates == current.coordinates && inputs.scalars == current.scalars
            && inputs.functions == current.functions && inputs.metric == current.metric
    }
    func applying(to current: CalculationInputs) -> CalculationInputs {
        var selected = inputs
        selected.outputs = current.outputs
        selected.metricSign = current.metricSign
        selected.riemannSign = current.riemannSign
        return selected
    }
    static let all: [MetricPreset] = [
        .init(id: "flat", name: "Flat spacetime", detail: "Spherical coordinates", inputs: .init()),
        .init(id: "schwarzschild", name: "Schwarzschild", detail: "Mass M", inputs: .init(scalars: "M", metric:
            "[-(1 - 2*M/r), 0, 0, 0],\n[0, 1/(1 - 2*M/r), 0, 0],\n[0, 0, r^2, 0],\n[0, 0, 0, r^2*sin(theta)^2]")),
        .init(id: "rn", name: "Reissner–Nordström", detail: "Mass M, charge Q", inputs: .init(scalars: "M, Q", metric:
            "[-(1 - 2*M/r + Q^2/r^2), 0, 0, 0],\n[0, 1/(1 - 2*M/r + Q^2/r^2), 0, 0],\n[0, 0, r^2, 0],\n[0, 0, 0, r^2*sin(theta)^2]")),
        .init(id: "kerr", name: "Kerr", detail: "Mass M, spin a", inputs: .init(scalars: "M, a", metric:
            "[-(1 - 2*M*r/(r^2 + a^2*cos(theta)^2)), 0, 0, -2*M*a*r*sin(theta)^2/(r^2 + a^2*cos(theta)^2)],\n[0, (r^2 + a^2*cos(theta)^2)/(r^2 - 2*M*r + a^2), 0, 0],\n[0, 0, r^2 + a^2*cos(theta)^2, 0],\n[-2*M*a*r*sin(theta)^2/(r^2 + a^2*cos(theta)^2), 0, 0, (r^2 + a^2 + 2*M*a^2*r*sin(theta)^2/(r^2 + a^2*cos(theta)^2))*sin(theta)^2]")),
        .init(id: "flrw", name: "FLRW", detail: "Scale factor a(t), curvature k", inputs: .init(scalars: "k", functions: "a(t)", metric:
            "[-1, 0, 0, 0],\n[0, a(t)^2/(1 - k*r^2), 0, 0],\n[0, 0, a(t)^2*r^2, 0],\n[0, 0, 0, a(t)^2*r^2*sin(theta)^2]")),
        .init(id: "sphere", name: "Two-sphere", detail: "Radius L", inputs: .init(coordinates: "theta, phi", scalars: "L", metric:
            "[L^2, 0],\n[0, L^2*sin(theta)^2]"))
    ]
}
