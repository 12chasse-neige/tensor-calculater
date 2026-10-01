import Foundation

/// Give each contiguous group of upper/lower indices its own horizontal slot.
/// In particular, the lower indices of R^mu{}_{nu rho sigma} start to the right
/// of mu, rather than sharing its vertical column. Do not reorder mixed slots.
enum TensorNotation {
    static func indexed(_ symbol: String, labels: [String], variance: String) -> String {
        let positions = Array(variance)
        var runs: [(upper: Bool, labels: [String])] = []
        for (index, label) in labels.enumerated() {
            let upper = positions.indices.contains(index) && positions[index] == "u"
            if let last = runs.last, last.upper == upper {
                runs[runs.count - 1].labels.append(label)
            } else {
                runs.append((upper, [label]))
            }
        }
        return symbol + runs.enumerated().map { index, run in
            (index == 0 ? "" : "{}") + (run.upper ? "^{" : "_{")
                + run.labels.joined(separator: "\\,") + "}"
        }.joined()
    }

    static let christoffel = #"\Gamma^{\alpha}{}_{\beta\,\gamma}"#
    static let riemann = #"R^{\mu}{}_{\nu\,\rho\,\sigma}"#
    static let ricciContraction = #"R_{\nu\,\sigma}=R^{\mu}{}_{\nu\,\mu\,\sigma}"#
}

enum RiemannConvention: Int, CaseIterable, Identifiable {
    // Preserve the original worker's default definition.
    case derivativeLastFirst = 1
    case derivativeFirstFirst = -1
    var id: Int { rawValue }
    var accessibilityLabel: String {
        self == .derivativeLastFirst ? "Derivative sigma first" : "Derivative rho first"
    }
    var buttonLatex: String {
        self == .derivativeLastFirst
            ? #"\partial_{\sigma}\Gamma^{\mu}{}_{\rho\,\nu}-\partial_{\rho}\Gamma^{\mu}{}_{\sigma\,\nu}"#
            : #"\partial_{\rho}\Gamma^{\mu}{}_{\sigma\,\nu}-\partial_{\sigma}\Gamma^{\mu}{}_{\rho\,\nu}"#
    }
    var definitionLatex: String {
        TensorNotation.riemann + "=" + buttonLatex + (self == .derivativeLastFirst
            ? #"+\Gamma^{\mu}{}_{\sigma\,\lambda}\Gamma^{\lambda}{}_{\rho\,\nu}-\Gamma^{\mu}{}_{\rho\,\lambda}\Gamma^{\lambda}{}_{\sigma\,\nu}"#
            : #"+\Gamma^{\mu}{}_{\rho\,\lambda}\Gamma^{\lambda}{}_{\sigma\,\nu}-\Gamma^{\mu}{}_{\sigma\,\lambda}\Gamma^{\lambda}{}_{\rho\,\nu}"#)
    }
    init?(resultConvention: String) {
        if resultConvention.hasPrefix("R^rho_{sigma mu nu} = d_mu") { self = .derivativeFirstFirst }
        else if resultConvention.hasPrefix("R^rho_{sigma mu nu} = d_nu") { self = .derivativeLastFirst }
        else { return nil }
    }
}

/// All sidebar rows have explicit tags. Selecting a preset both highlights that
/// row and opens its inputs; changing conventions does not select another row.
enum CalculationNavigation {
    static func presetKey(_ id: String) -> String { "preset:" + id }
    static func showsInputs(_ selection: String?) -> Bool {
        selection == nil || selection == "input" || selection?.hasPrefix("preset:") == true
    }
    static func preset(for selection: String?) -> MetricPreset? {
        MetricPreset.all.first { presetKey($0.id) == selection }
    }
}
