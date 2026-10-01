import Foundation

/// A deliberately small LaTeX input grammar. Unknown commands are rejected,
/// never silently discarded. The C++ parser still validates every expression.
enum MetricNotation {
    struct InputError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }
    static func expression(_ source: String, depth: Int = 0) throws -> String {
        guard depth < 64, source.utf8.count < 100_000 else { throw InputError(message: "Metric expression is too large or deeply nested.") }
        let chars = Array(source)
        var i = 0
        var result = ""
        func group() throws -> String {
            while i < chars.count && chars[i].isWhitespace { i += 1 }
            guard i < chars.count, chars[i] == "{" else { throw InputError(message: "Use braces for LaTeX arguments, for example \\frac{1}{r} or \\sin{theta}.") }
            i += 1
            let start = i
            var level = 1
            while i < chars.count {
                if chars[i] == "{" { level += 1 }
                if chars[i] == "}" { level -= 1 }
                if level == 0 { let text = String(chars[start..<i]); i += 1; return try expression(text, depth: depth + 1) }
                i += 1
            }
            throw InputError(message: "Unclosed LaTeX argument.")
        }
        let symbols = Set("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau upsilon phi chi psi omega".split(separator: " ").map(String.init))
        let functions = Set("sin cos tan cot sec csc sinh cosh tanh exp log ln asin acos atan".split(separator: " ").map(String.init))
        while i < chars.count {
            let c = chars[i]
            if c == "{" { result += "(" + (try group()) + ")"; continue }
            if c == "}" { throw InputError(message: "Unexpected closing brace.") }
            guard c == "\\" else { result.append(c); i += 1; continue }
            i += 1
            let start = i
            while i < chars.count && chars[i].isLetter { i += 1 }
            let command = String(chars[start..<i])
            if command.isEmpty {
                guard i < chars.count, [",", ";", "!", " "].contains(chars[i]) else { throw InputError(message: "Unsupported LaTeX escape.") }
                result += " "; i += 1
            } else if command == "frac" || command == "dfrac" || command == "tfrac" {
                let numerator = try group(), denominator = try group()
                result += "((" + numerator + ")/(" + denominator + "))"
            } else if command == "sqrt" { result += "sqrt(" + (try group()) + ")" }
            else if command == "left" || command == "right" { continue }
            else if command == "cdot" || command == "times" { result += "*" }
            else if symbols.contains(command) { result += " " + command + " " }
            else if functions.contains(command) {
                // TeX function powers precede their argument: \sin^2{theta}.
                while i < chars.count && chars[i].isWhitespace { i += 1 }
                var exponent: String?
                if i < chars.count && chars[i] == "^" {
                    i += 1
                    if i < chars.count && chars[i] == "{" { exponent = try group() }
                    else if i < chars.count && chars[i].isNumber { exponent = String(chars[i]); i += 1 }
                    else { throw InputError(message: "Use braces around a function exponent.") }
                }
                while i < chars.count && chars[i].isWhitespace { i += 1 }
                if String(chars[i...]).hasPrefix(#"\left("#) { i += 5 }
                var argument: String
                if i < chars.count && chars[i] == "{" { argument = try group() }
                else if i < chars.count && chars[i] == "(" {
                    i += 1
                    let start = i
                    var level = 1
                    while i < chars.count {
                        if chars[i] == "(" { level += 1 }
                        if chars[i] == ")" { level -= 1 }
                        if level == 0 { break }
                        i += 1
                    }
                    guard i < chars.count else { throw InputError(message: "Unclosed function argument.") }
                    argument = try expression(String(chars[start..<i]), depth: depth + 1)
                    i += 1
                } else { throw InputError(message: "Enclose the argument of " + command + " in braces or parentheses.") }
                result += command + "(" + argument + ")" + (exponent.map { "^(" + $0 + ")" } ?? "")
            } else { throw InputError(message: "Unsupported LaTeX command: \\" + command) }
        }
        return result
    }

    static func rows(_ source: String) throws -> [[String]] {
        guard source.utf8.count < 100_000 else { throw InputError(message: "Metric input must be smaller than 100 KB.") }
        var text = source.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix("\\begin{") {
            for environment in ["pmatrix", "bmatrix", "matrix"] {
                let begin = "\\begin{\(environment)}", end = "\\end{\(environment)}"
                if text.hasPrefix(begin), text.hasSuffix(end) {
                    text = String(text.dropFirst(begin.count).dropLast(end.count))
                    var rows = text.components(separatedBy: "\\\\")
                    if rows.last?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true { rows.removeLast() }
                    return try rows.map { try $0.components(separatedBy: "&").map { try expression($0) } }
                }
            }
            throw InputError(message: "Use a complete matrix, pmatrix, or bmatrix environment.")
        }
        if text.hasPrefix("diag("), text.hasSuffix(")") {
            let entries = split(String(text.dropFirst(5).dropLast()))
            return entries.indices.map { row in entries.indices.map { row == $0 ? entries[row] : "0" } }
        }
        if text.hasPrefix("Matrix(") && text.hasSuffix(")") { text = String(text.dropFirst(7).dropLast()) }
        if text.replacingOccurrences(of: " ", with: "").hasPrefix("[[") { text = String(text.dropFirst().dropLast()) }
        let matches = try NSRegularExpression(pattern: #"\[([^\[\]]*)\]"#).matches(in: text, range: NSRange(text.startIndex..., in: text))
        guard !matches.isEmpty else { throw InputError(message: "Enter bracketed matrix rows or a LaTeX matrix.") }
        var remainder = text
        for match in matches.reversed() { if let range = Range(match.range, in: remainder) { remainder.removeSubrange(range) } }
        guard remainder.allSatisfy({ $0.isWhitespace || $0 == "," }) else { throw InputError(message: "Unexpected text outside metric rows.") }
        return try matches.map { match in try split(String(text[Range(match.range(at: 1), in: text)!])).map { try expression($0) } }
    }
    private static func split(_ text: String) -> [String] {
        var depth = 0, current = "", values: [String] = []
        for c in text {
            if c == "(" || c == "{" { depth += 1 }
            if c == ")" || c == "}" { depth -= 1 }
            if c == "," && depth == 0 { values.append(current); current = "" } else { current.append(c) }
        }
        values.append(current)
        return values
    }
    static func plain(_ source: String) throws -> String {
        try rows(source).map { "[" + $0.joined(separator: ", ") + "]" }.joined(separator: ",\n")
    }
    static func latex(_ source: String) throws -> String {
        if source.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("\\begin{") { _ = try rows(source); return source }
        let body = try rows(source).map { row in row.map { entry in
            var value = entry.replacingOccurrences(of: "**", with: "^")
            value = groupedPowers(value)
            value = value.replacingOccurrences(of: #"\^(-?(?:[0-9]+(?:\.[0-9]+)?|[A-Za-z_][A-Za-z_0-9]*))"#, with: "^{$1}", options: .regularExpression)
            value = value.replacingOccurrences(of: #"\b(theta|phi|alpha|beta|gamma|lambda|mu|nu|rho|sigma|omega|pi|sin|cos|tan|exp|log)\b"#, with: #"\\$1"#, options: .regularExpression)
            return value.replacingOccurrences(of: "*", with: " ")
        }.joined(separator: " & ") }.joined(separator: " \\\\ ")
        return "\\begin{pmatrix}" + body + "\\end{pmatrix}"
    }
    private static func groupedPowers(_ text: String) -> String {
        let chars = Array(text)
        var i = 0, result = ""
        while i < chars.count {
            if chars[i] == "^", i + 1 < chars.count, chars[i + 1] == "(" {
                let start = i + 2
                var end = start, depth = 1
                while end < chars.count {
                    if chars[end] == "(" { depth += 1 }
                    if chars[end] == ")" { depth -= 1 }
                    if depth == 0 { break }
                    end += 1
                }
                if end < chars.count {
                    result += "^{" + groupedPowers(String(chars[start..<end])) + "}"
                    i = end + 1
                    continue
                }
            }
            result.append(chars[i]); i += 1
        }
        return result
    }
    static func negated(_ source: String) throws -> String {
        try rows(source).map { "[" + $0.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) == "0" ? "0" : "-(\($0))" }.joined(separator: ", ") + "]" }.joined(separator: ",\n")
    }
}
