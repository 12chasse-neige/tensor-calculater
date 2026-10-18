import Foundation

extension TensorResult {
    /// Export the complete stored result, independent of the component search.
    /// Unresolved symbolic zeros remain included, just as in the result browser.
    func latexBlock(coordinates: [String]) -> String {
        let equations: [String]
        if components.isEmpty {
            let indices = (0..<rank).map { "i_{\($0 + 1)}" }
            equations = [TensorNotation.indexed(symbol, labels: indices, variance: variance) + " &= 0"]
        } else {
            equations = components.map {
                label(for: $0, coordinates: coordinates) + " &= " + $0.latex
            }
        }
        return "\\[\n\\begin{aligned}\n" + equations.joined(separator: " \\\\\n") + "\n\\end{aligned}\n\\]"
    }
}

extension ScalarResult {
    var latexBlock: String { "\\[\n" + symbol + " = " + latex + "\n\\]" }
}
