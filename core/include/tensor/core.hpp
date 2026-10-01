#pragma once

#include <symengine/basic.h>
#include <symengine/symbol.h>
#include <nlohmann/json.hpp>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tensor {
using Expr = SymEngine::RCP<const SymEngine::Basic>;
using Symbol = SymEngine::RCP<const SymEngine::Symbol>;
using Index = std::vector<int>;
using Components = std::map<Index, Expr>;
using JSON = nlohmann::json;
using Progress = std::function<void(const std::string&, const std::string&)>;

struct FunctionDefinition { std::string name; std::vector<std::string> arguments; };
struct Input {
    int riemann_sign = 1;
    std::vector<std::string> names;
    std::vector<Symbol> coordinates;
    std::map<std::string, Expr> symbols;
    std::map<std::string, FunctionDefinition> functions;
    std::vector<Expr> metric; // row-major g_{mu nu}, exact symbolic expressions
    std::set<std::string> outputs;
};

// The normalizer applies algebraic identities only. Numerical samples never
// establish a symbolic zero, and unresolved expressions remain in the result.
Expr normalize(const Expr& expression);
Expr cancel_denominator_factors(const Expr& expanded_numerator, const Expr& factored_denominator);
bool exact_zero(const Expr& expression);
void check_cancelled();
void request_cancel();
bool cancellation_requested();
void validate_finite(const Expr& expression);
std::string expression_latex(const Expr& expression);
std::vector<std::string> split_csv(const std::string& text);
Expr parse_expression(const std::string& text, const Input& input);
Input parse_input(const JSON& request);
JSON calculate(const Input& input, const Progress& progress);
} // namespace tensor
