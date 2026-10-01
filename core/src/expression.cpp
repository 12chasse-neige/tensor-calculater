#include "tensor/core.hpp"
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/functions.h>
#include <symengine/integer.h>
#include <symengine/infinity.h>
#include <symengine/nan.h>
#include <symengine/number.h>
#include <symengine/printers.h>
#include <symengine/simplify.h>
#include <symengine/subs.h>
#include <symengine/visitor.h>
#include <csignal>
#include <stdexcept>

namespace tensor {
namespace se = SymEngine;
namespace {
volatile std::sig_atomic_t cancelled = 0;

size_t complexity(const Expr& expression, size_t limit = 2500) {
    size_t count = 1;
    for (const auto& arg : expression->get_args()) {
        if (count >= limit) break;
        count += complexity(arg, limit - count);
    }
    return count;
}

// Estimate polynomial expansion growth before calling expand. An expression
// budget limits display simplification, never the underlying exact calculation.
size_t expansion_size(const Expr& e, size_t cap = 6000) {
    if (se::is_a<se::Add>(*e)) {
        size_t sum = 0;
        for (const auto& a : e->get_args()) sum = std::min(cap, sum + expansion_size(a, cap));
        return sum;
    }
    if (se::is_a<se::Mul>(*e)) {
        size_t product = 1;
        for (const auto& a : e->get_args()) product = std::min(cap, product * expansion_size(a, cap));
        return product;
    }
    if (se::is_a<se::Pow>(*e)) {
        const auto& p = se::down_cast<const se::Pow&>(*e);
        if (se::is_a<se::Integer>(*p.get_exp())) {
            auto exponent = se::down_cast<const se::Integer&>(*p.get_exp()).as_int();
            if (exponent > 0 && exponent <= 12) {
                size_t product = 1, base = expansion_size(p.get_base(), cap);
                for (long i = 0; i < exponent; ++i) product = std::min(cap, product * base);
                return product;
            }
            if (exponent > 12 && expansion_size(p.get_base(), cap) > 1) return cap;
        }
    }
    return 1;
}

void trig_replacements(const Expr& e, se::map_basic_basic& replacements) {
    if (se::is_a<se::Pow>(*e)) {
        const auto& p = se::down_cast<const se::Pow&>(*e);
        if (se::is_a<se::Cos>(*p.get_base()) && se::is_a<se::Integer>(*p.get_exp())) {
            auto exponent = se::down_cast<const se::Integer&>(*p.get_exp()).as_int();
            if (exponent > 0 && exponent <= 12 && exponent % 2 == 0) {
                const auto& c = se::down_cast<const se::Cos&>(*p.get_base());
                replacements[e] = se::pow(se::sub(se::one, se::pow(se::sin(c.get_arg()), 2)), exponent / 2);
            }
        }
    }
    if (se::is_a<se::Tan>(*e) || se::is_a<se::Cot>(*e) || se::is_a<se::Sec>(*e) || se::is_a<se::Csc>(*e)) {
        auto arg = se::down_cast<const se::OneArgFunction&>(*e).get_arg();
        if (se::is_a<se::Tan>(*e)) replacements[e] = se::div(se::sin(arg), se::cos(arg));
        else if (se::is_a<se::Cot>(*e)) replacements[e] = se::div(se::cos(arg), se::sin(arg));
        else if (se::is_a<se::Sec>(*e)) replacements[e] = se::div(se::one, se::cos(arg));
        else replacements[e] = se::div(se::one, se::sin(arg));
    }
    for (const auto& arg : e->get_args()) trig_replacements(arg, replacements);
}

Expr polynomial_normalize(const Expr& e) {
    if (expansion_size(e) >= 6000) return e;
    auto expanded = se::expand(e);
    se::map_basic_basic replacements;
    trig_replacements(expanded, replacements);
    if (!replacements.empty()) {
        auto rewritten = se::subs(expanded, replacements);
        if (expansion_size(rewritten) < 6000) expanded = se::expand(rewritten);
    }
    return expanded;
}
} // namespace

void request_cancel() { cancelled = 1; }
bool cancellation_requested() { return cancelled != 0; }
void check_cancelled() { if (cancelled) throw std::runtime_error("Calculation cancelled."); }
bool exact_zero(const Expr& expression) { return se::eq(*expression, *se::zero); }
void validate_finite(const Expr& expression) {
    if (se::is_a<se::Infty>(*expression) || se::is_a<se::NaN>(*expression))
        throw std::runtime_error("Expression contains an undefined or infinite value.");
    for (const auto& arg : expression->get_args()) validate_finite(arg);
}

Expr normalize(const Expr& expression) {
    check_cancelled();
    if (exact_zero(expression)) return se::zero;
    // Large expressions remain exact; avoid an unbounded heuristic search.
    if (complexity(expression) >= 2500) return expression;
    auto simplified = se::simplify(expression);
    Expr numerator, denominator;
    se::as_numer_denom(simplified, se::outArg(numerator), se::outArg(denominator));
    numerator = polynomial_normalize(numerator);
    if (exact_zero(numerator)) return se::zero;
    // Cancel only factors proved to divide the numerator exactly over Q[atoms].
    // Preserve denominator factors instead of expanding them into a large sum.
    if (exact_zero(polynomial_normalize(denominator)))
        throw std::runtime_error("Expression denominator is identically zero.");
    auto candidate = cancel_denominator_factors(numerator, denominator);
    // Proof of zero above is retained even if a larger nonzero display form is
    // rejected. This is deliberately independent of numeric tolerance.
    if (complexity(candidate) <= complexity(simplified) * 2 + 20) return candidate;
    return simplified;
}

std::string expression_latex(const Expr& expression) {
    return se::latex(*expression);
}
} // namespace tensor
