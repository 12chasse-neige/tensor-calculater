#include "tensor/core.hpp"
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/functions.h>
#include <symengine/integer.h>
#include <symengine/constants.h>
#include <symengine/derivative.h>
#include <symengine/visitor.h>
#include <cctype>
#include <regex>
#include <stdexcept>

namespace tensor {
namespace se = SymEngine;
namespace {
std::string trim(const std::string& s) {
    auto first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
const std::set<std::string> reserved = {
    "sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "exp", "sqrt", "log", "ln", "Abs", "abs",
    "Rational", "Integer", "Float", "S", "pi", "E", "I", "diff",
    "Derivative", "Matrix", "diag", "eye", "zeros", "ones"
};
void identifier(const std::string& name) {
    static const std::regex valid("^[A-Za-z_][A-Za-z_0-9]*$");
    if (!std::regex_match(name, valid) || reserved.count(name))
        throw std::runtime_error("Invalid or reserved symbol name: " + name);
}

// A restricted mathematical grammar, not an evaluator for another language.
// Every identifier comes from declared coordinates, constants or functions.
class Parser {
    enum class Kind { End, Number, Name, Plus, Minus, Star, Slash, Power, Left, Right, Comma };
    struct Token { Kind kind; std::string text; size_t offset; };
    std::string text;
    const Input& input;
    size_t position = 0;
    size_t tokens = 0;
    int depth = 0;
    Token token;

    [[noreturn]] void error(const std::string& message) const {
        throw std::runtime_error(message + " at character " + std::to_string(token.offset + 1) + ".");
    }
    void next() {
        check_cancelled();
        if (++tokens > 20000) error("Expression contains too many tokens");
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        token = {Kind::End, "", position};
        if (position == text.size()) return;
        char c = text[position++];
        switch (c) {
            case '+': token.kind = Kind::Plus; break;
            case '-': token.kind = Kind::Minus; break;
            case '*':
                if (position < text.size() && text[position] == '*') { ++position; token.kind = Kind::Power; }
                else token.kind = Kind::Star;
                break;
            case '/': token.kind = Kind::Slash; break;
            case '^': token.kind = Kind::Power; break;
            case '(': token.kind = Kind::Left; break;
            case ')': token.kind = Kind::Right; break;
            case ',': token.kind = Kind::Comma; break;
            default:
                if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                    token.kind = Kind::Name;
                    while (position < text.size() && (std::isalnum(static_cast<unsigned char>(text[position])) || text[position] == '_')) ++position;
                } else if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
                    token.kind = Kind::Number;
                    while (position < text.size() && (std::isdigit(static_cast<unsigned char>(text[position])) || text[position] == '.')) ++position;
                    if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) {
                        size_t start = position++;
                        if (position < text.size() && (text[position] == '+' || text[position] == '-')) ++position;
                        size_t digits = position;
                        while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position]))) ++position;
                        if (digits == position) position = start; // 2E is implicit multiplication by Euler's number.
                    }
                } else error("Unexpected character");
        }
        token.text = text.substr(token.offset, position - token.offset);
    }
    Expr number(const std::string& value) {
        if (value.size() > 1024) error("Number contains too many digits");
        static const std::regex valid("^([0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE]([+-]?[0-9]+))?$");
        std::smatch match;
        if (!std::regex_match(value, match, valid)) error("Invalid number");
        std::string mantissa = match[1];
        int exponent = match[2].matched ? std::stoi(match[2]) : 0;
        if (exponent < -1000 || exponent > 1000) error("Numeric exponent is too large");
        auto dot = mantissa.find('.');
        int decimals = dot == std::string::npos ? 0 : static_cast<int>(mantissa.size() - dot - 1);
        if (dot != std::string::npos) mantissa.erase(dot, 1);
        auto numerator = se::integer(se::integer_class(mantissa));
        auto power = se::pow(se::integer(10), se::integer(std::abs(exponent - decimals)));
        return exponent >= decimals ? se::mul(numerator, power) : se::div(numerator, power);
    }
    Expr sum() {
        auto left = product();
        while (token.kind == Kind::Plus || token.kind == Kind::Minus) {
            auto op = token.kind; next(); auto right = product();
            left = op == Kind::Plus ? se::add(left, right) : se::sub(left, right);
        }
        return left;
    }
    Expr product() {
        auto left = unary();
        while (true) {
            if (token.kind == Kind::Star || token.kind == Kind::Slash) {
                auto op = token.kind; next(); auto right = unary();
                if (op == Kind::Slash && exact_zero(right)) error("Division by zero");
                left = op == Kind::Star ? se::mul(left, right) : se::div(left, right);
            } else if (token.kind == Kind::Name || token.kind == Kind::Number || token.kind == Kind::Left) {
                left = se::mul(left, unary()); // e.g. 2M/r or 2(r+1)
            } else break;
        }
        return left;
    }
    Expr unary() {
        struct DepthGuard { int& d; explicit DepthGuard(int& value): d(value) { ++d; } ~DepthGuard() { --d; } } guard(depth);
        if (depth > 128) error("Expression nesting is too deep");
        if (token.kind == Kind::Plus) { next(); return unary(); }
        if (token.kind == Kind::Minus) { next(); return se::neg(unary()); }
        return power();
    }
    Expr power() {
        auto left = primary();
        if (token.kind == Kind::Power) {
            next(); auto exponent = unary();
            if (se::is_a<se::Integer>(*exponent)) {
                const auto value = se::down_cast<const se::Integer&>(*exponent).as_integer_class();
                if (value > 1000 || value < -1000) error("Integer power must be between -1000 and 1000");
                if (exact_zero(left) && value < 0) error("Division by zero");
                if (se::is_a_Number(*left)) {
                    auto count = se::down_cast<const se::Integer&>(*exponent).as_int();
                    if (left->__str__().size() * static_cast<size_t>(std::abs(count)) > 10000)
                        error("Numeric power would create more than 10000 digits");
                }
            }
            left = se::pow(left, exponent);
        }
        return left;
    }
    Expr function(const std::string& name, const std::vector<Expr>& args) {
        auto arity = [&](size_t n) { if (args.size() != n) error(name + " requires " + std::to_string(n) + " argument(s)"); };
        if (name == "diff" || name == "Derivative") {
            if (args.size() < 2) error("diff requires an expression and a coordinate");
            auto value = args[0];
            for (size_t i = 1; i < args.size(); ++i) {
                if (!se::is_a<se::Symbol>(*args[i])) error("Derivative variable must be a coordinate");
                auto variable = se::rcp_static_cast<const se::Symbol>(args[i]);
                if (std::find(input.names.begin(), input.names.end(), variable->get_name()) == input.names.end()) error("Derivative variable is not a coordinate");
                long count = 1;
                if (i + 1 < args.size() && se::is_a<se::Integer>(*args[i+1])) count = se::down_cast<const se::Integer&>(*args[++i]).as_int();
                if (count < 0 || count > 12) error("Derivative order must be between 0 and 12");
                for (long j = 0; j < count; ++j) { check_cancelled(); value = value->diff(variable); }
            }
            return value;
        }
        if (name == "Rational") { arity(2); if (exact_zero(args[1])) error("Division by zero"); return se::div(args[0], args[1]); }
        if (name == "Integer" || name == "Float" || name == "S") { arity(1); return args[0]; }
        auto custom = input.functions.find(name);
        if (custom != input.functions.end()) {
            arity(custom->second.arguments.size());
            // Declared dependencies constrain names, but composed arguments
            // (a(2*t), for example) remain useful and differentiate correctly.
            std::set<std::string> dependencies(custom->second.arguments.begin(), custom->second.arguments.end());
            for (const auto& arg : args) {
                for (const auto& symbol : se::free_symbols(*arg)) {
                    const auto& n = se::down_cast<const se::Symbol&>(*symbol).get_name();
                    if (input.symbols.count(n) && std::find(input.names.begin(), input.names.end(), n) != input.names.end() && !dependencies.count(n)) error("Function argument is outside its declared coordinate dependencies");
                }
            }
            return se::function_symbol(name, args);
        }
        arity(1);
        auto a = args[0];
        if (name == "sin") return se::sin(a);
        if (name == "cos") return se::cos(a);
        if (name == "tan") return se::tan(a);
        if (name == "cot") return se::cot(a);
        if (name == "sec") return se::sec(a);
        if (name == "csc") return se::csc(a);
        if (name == "asin") return se::asin(a);
        if (name == "acos") return se::acos(a);
        if (name == "atan") return se::atan(a);
        if (name == "sinh") return se::sinh(a);
        if (name == "cosh") return se::cosh(a);
        if (name == "tanh") return se::tanh(a);
        if (name == "exp") return se::exp(a);
        if (name == "sqrt") return se::sqrt(a);
        if (name == "log" || name == "ln") return se::log(a);
        if (name == "Abs" || name == "abs") return se::abs(a);
        error("Undefined function: " + name);
    }
    Expr primary() {
        if (token.kind == Kind::Number) { auto value = token.text; next(); return number(value); }
        if (token.kind == Kind::Left) {
            next(); auto value = sum(); if (token.kind != Kind::Right) error("Expected closing parenthesis"); next(); return value;
        }
        if (token.kind == Kind::Name) {
            auto name = token.text; next();
            // A declared scalar next to parentheses means multiplication.
            if (token.kind == Kind::Left && !input.symbols.count(name) && name != "pi" && name != "E" && name != "I") {
                next(); std::vector<Expr> args;
                if (token.kind != Kind::Right) {
                    args.push_back(sum());
                    while (token.kind == Kind::Comma) { next(); args.push_back(sum()); }
                }
                if (token.kind != Kind::Right) error("Expected closing parenthesis");
                next(); return function(name, args);
            }
            if (name == "pi") return se::pi;
            if (name == "E") return se::E;
            if (name == "I") return se::I;
            auto found = input.symbols.find(name);
            if (found == input.symbols.end()) error("Undefined symbol: " + name + "; declare it as a constant or function");
            return found->second;
        }
        error("Expected a mathematical expression");
    }
public:
    Parser(std::string expression, const Input& i) : text(std::move(expression)), input(i) { next(); }
    Expr parse() { auto value = sum(); if (token.kind != Kind::End) error("Unexpected token"); return value; }
};

std::vector<std::vector<std::string>> matrix_rows(std::string text) {
    text = trim(text);
    if (text.starts_with("Matrix(")) {
        if (!text.ends_with(')')) throw std::runtime_error("Unclosed Matrix constructor.");
        text = trim(text.substr(7, text.size()-8));
    }
    // Detect an enclosing list from its second non-whitespace character.
    if (!text.empty() && text[0] == '[') {
        size_t second = text.find_first_not_of(" \t\r\n", 1);
        if (second != std::string::npos && text[second] == '[') {
            if (text.back() != ']') throw std::runtime_error("Unclosed metric matrix.");
            text = trim(text.substr(1, text.size()-2));
        }
    }
    std::vector<std::vector<std::string>> rows;
    size_t p = 0;
    while (p < text.size()) {
        while (p < text.size() && std::isspace(static_cast<unsigned char>(text[p]))) ++p;
        if (p == text.size()) break;
        if (text[p] != '[') throw std::runtime_error("Each metric row must be enclosed in square brackets.");
        size_t start = ++p;
        int depth = 1;
        while (p < text.size() && depth) { if (text[p] == '[') ++depth; if (text[p] == ']') --depth; ++p; }
        if (depth) throw std::runtime_error("Unclosed metric row.");
        rows.push_back(split_csv(text.substr(start, p-start-1)));
        size_t after = p;
        while (p < text.size() && std::isspace(static_cast<unsigned char>(text[p]))) ++p;
        if (p < text.size() && text[p] == ',') {
            ++p;
            size_t next = text.find_first_not_of(" \t\r\n", p);
            if (next == std::string::npos) throw std::runtime_error("Trailing comma after metric row.");
        } else if (p < text.size() && text.substr(after,p-after).find('\n') == std::string::npos) {
            throw std::runtime_error("Separate metric rows with a comma or newline.");
        }
    }
    return rows;
}
} // namespace

std::vector<std::string> split_csv(const std::string& text) {
    if (trim(text).empty()) return {};
    std::vector<std::string> values;
    std::vector<char> stack;
    size_t start = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c == '(' || c == '[') stack.push_back(c);
        else if (c == ')' || c == ']') {
            if (stack.empty() || (c == ')' ? stack.back() != '(' : stack.back() != '[')) throw std::runtime_error("Mismatched brackets.");
            stack.pop_back();
        } else if (c == ',' && stack.empty()) {
            auto value = trim(text.substr(start, i-start));
            if (value.empty()) throw std::runtime_error("Empty item in comma-separated input.");
            values.push_back(value); start = i+1;
        }
    }
    if (!stack.empty()) throw std::runtime_error("Unclosed brackets.");
    auto last = trim(text.substr(start));
    if (last.empty()) throw std::runtime_error("Empty item in comma-separated input.");
    values.push_back(last);
    return values;
}

Expr parse_expression(const std::string& text, const Input& input) {
    auto value = Parser(text, input).parse();
    validate_finite(value);
    return value;
}

Input parse_input(const JSON& request) {
    if (!request.is_object() || request.value("version", 0) != 1) throw std::runtime_error("Unsupported protocol version; expected version 1.");
    Input input;
    input.riemann_sign = request.value("riemann_sign", 1);
    if (input.riemann_sign != 1 && input.riemann_sign != -1) throw std::runtime_error("Riemann sign must be +1 or -1.");
    input.names = split_csv(request.at("coordinates").get<std::string>());
    if (input.names.empty() || input.names.size() > 8) throw std::runtime_error("Enter between 1 and 8 coordinates.");
    for (const auto& name : input.names) {
        identifier(name);
        if (input.symbols.count(name)) throw std::runtime_error("Duplicate coordinate: " + name);
        auto symbol = se::symbol(name); input.coordinates.push_back(symbol); input.symbols[name] = symbol;
    }
    for (const auto& name : split_csv(request.value("scalars", std::string{}))) {
        identifier(name);
        if (input.symbols.count(name)) throw std::runtime_error("Duplicate or conflicting constant: " + name);
        input.symbols[name] = se::symbol(name);
    }
    static const std::regex declaration("^([A-Za-z_][A-Za-z_0-9]*)\\s*\\((.*)\\)$");
    for (const auto& item : split_csv(request.value("functions", std::string{}))) {
        std::smatch match;
        if (!std::regex_match(item, match, declaration)) throw std::runtime_error("Declare functions using name(coordinates), for example a(t).");
        std::string name = match[1]; identifier(name);
        if (input.symbols.count(name) || input.functions.count(name)) throw std::runtime_error("Conflicting function name: " + name);
        auto args = split_csv(match[2]);
        if (args.empty()) throw std::runtime_error("A function needs at least one coordinate argument.");
        std::set<std::string> seen;
        for (const auto& arg : args) {
            if (std::find(input.names.begin(), input.names.end(), arg) == input.names.end() || !seen.insert(arg).second)
                throw std::runtime_error("Function arguments must be distinct declared coordinates.");
        }
        input.functions[name] = {name, args};
    }
    static const std::set<std::string> keys = {"inverse_metric", "christoffel", "riemann", "ricci", "ricci_scalar", "kretschmann"};
    if (!request.contains("outputs")) input.outputs = keys;
    else {
        for (const auto& key : request.at("outputs")) {
            auto name = key.get<std::string>();
            if (!keys.count(name)) throw std::runtime_error("Unknown calculation output: " + name);
            input.outputs.insert(name);
        }
    }
    if (input.outputs.empty()) throw std::runtime_error("Select at least one calculation output.");
    std::string metric_text = trim(request.at("metric").get<std::string>());
    const size_t n = input.names.size();
    if (metric_text.starts_with("diag(") && metric_text.ends_with(')')) {
        auto entries = split_csv(metric_text.substr(5, metric_text.size()-6));
        if (entries.size() != n) throw std::runtime_error("Metric size must match the coordinate count.");
        input.metric.assign(n*n, se::zero);
        for (size_t i = 0; i < n; ++i) input.metric[i*n+i] = parse_expression(entries[i],input);
    } else {
        auto rows = matrix_rows(metric_text);
        if (rows.size() != n) throw std::runtime_error("Metric row count must match the coordinate count.");
        for (size_t i = 0; i < n; ++i) {
            if (rows[i].size() != n) throw std::runtime_error("Metric must be square and match the coordinate count.");
            for (const auto& value : rows[i]) input.metric.push_back(parse_expression(value,input));
        }
    }
    for (size_t i = 0; i < n; ++i) for (size_t j = i+1; j < n; ++j) {
        if (!exact_zero(normalize(se::sub(input.metric[i*n+j],input.metric[j*n+i]))))
            throw std::runtime_error("Metric must be symmetric; entries (" + std::to_string(i) + "," + std::to_string(j) + ") and (" + std::to_string(j) + "," + std::to_string(i) + ") differ or could not be proved equal.");
    }
    return input;
}
} // namespace tensor
