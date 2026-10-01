#include "tensor/core.hpp"
#include <symengine/add.h>
#include <symengine/functions.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/number.h>
#include <symengine/pow.h>
#include <symengine/rational.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace tensor {
namespace se = SymEngine;
namespace {

// This is deliberately a known-factor canceller, not a general polynomial GCD
// implementation. Every accepted division is an exact identity over Q[atoms].
// Abandoning a calculation at any budget boundary preserves the original ratio.
constexpr size_t term_limit = 999;
constexpr size_t operation_limit = 99999;
constexpr size_t atom_limit = 128;
constexpr uint32_t degree_limit = 4096;
constexpr size_t coefficient_character_limit = 4096;

struct Limited {};

struct Budget {
    size_t operations = 0;
    void step(size_t count = 1) {
        check_cancelled();
        if (count > operation_limit - operations) throw Limited{};
        operations += count;
    }
};

// Sparse exponent vectors do not need resizing when another atom is discovered.
// Atom indices only increase; this keeps the ordering stable during conversion.
using Monomial = std::map<size_t, uint32_t>;
uint32_t degree(const Monomial& monomial) {
    uint32_t total = 0;
    for (const auto& [atom, exponent] : monomial) {
        (void)atom;
        if (exponent > degree_limit - total) throw Limited{};
        total += exponent;
    }
    return total;
}

struct GradedLex {
    bool operator()(const Monomial& left, const Monomial& right) const {
        auto ld = degree(left), rd = degree(right);
        if (ld != rd) return ld < rd;
        auto li = left.begin(), ri = right.begin();
        while (li != left.end() || ri != right.end()) {
            if (ri == right.end() || (li != left.end() && li->first < ri->first)) return false;
            if (li == left.end() || ri->first < li->first) return true;
            if (li->second != ri->second) return li->second < ri->second;
            ++li; ++ri;
        }
        return false;
    }
};
using Polynomial = std::map<Monomial, Expr, GradedLex>;

bool rational_coefficient(const Expr& expression) {
    return se::is_a<se::Integer>(*expression) || se::is_a<se::Rational>(*expression);
}

void bound_coefficient(const Expr& coefficient) {
    if (!rational_coefficient(coefficient) || coefficient->__str__().size() > coefficient_character_limit)
        throw Limited{};
}

void add_term(Polynomial& polynomial, const Monomial& monomial, const Expr& coefficient, Budget& budget) {
    budget.step();
    if (exact_zero(coefficient)) return;
    auto found = polynomial.find(monomial);
    auto combined = found == polynomial.end() ? coefficient : se::add(found->second, coefficient);
    bound_coefficient(combined);
    if (exact_zero(combined)) {
        if (found != polynomial.end()) polynomial.erase(found);
    } else {
        polynomial[monomial] = combined;
        if (polynomial.size() > term_limit) throw Limited{};
    }
}

Monomial multiply_monomials(const Monomial& left, const Monomial& right, Budget& budget) {
    budget.step(left.size() + right.size() + 1);
    auto product = left;
    for (const auto& [atom, exponent] : right) {
        auto& existing = product[atom];
        if (exponent > degree_limit - existing) throw Limited{};
        existing += exponent;
    }
    (void)degree(product);
    return product;
}

bool divide_monomials(const Monomial& numerator, const Monomial& denominator,
                      Monomial& quotient, Budget& budget) {
    budget.step(numerator.size() + denominator.size() + 1);
    quotient = numerator;
    for (const auto& [atom, exponent] : denominator) {
        auto found = quotient.find(atom);
        if (found == quotient.end() || found->second < exponent) return false;
        if (found->second == exponent) quotient.erase(found);
        else found->second -= exponent;
    }
    return true;
}

Polynomial multiply_polynomials(const Polynomial& left, const Polynomial& right, Budget& budget) {
    Polynomial result;
    for (const auto& [lm, lc] : left) for (const auto& [rm, rc] : right) {
        auto monomial = multiply_monomials(lm, rm, budget);
        budget.step();
        add_term(result, monomial, se::mul(lc, rc), budget);
    }
    return result;
}

Polynomial power_polynomial(Polynomial base, uint32_t exponent, Budget& budget) {
    Polynomial result{{Monomial{}, se::one}};
    while (exponent) {
        budget.step();
        if (exponent & 1U) result = multiply_polynomials(result, base, budget);
        exponent >>= 1U;
        if (exponent) base = multiply_polynomials(base, base, budget);
    }
    return result;
}

class Algebra {
    Budget& budget;
    std::map<Expr, size_t, se::RCPBasicKeyLess> atom_indices;
    std::vector<Expr> atoms;

    Polynomial atom(const Expr& expression) {
        auto found = atom_indices.find(expression);
        size_t index;
        if (found == atom_indices.end()) {
            if (atoms.size() >= atom_limit) throw Limited{};
            index = atoms.size(); atoms.push_back(expression); atom_indices[expression] = index;
        } else index = found->second;
        return {{Monomial{{index, 1}}, se::one}};
    }

public:
    explicit Algebra(Budget& b) : budget(b) {}

    Polynomial convert(const Expr& expression, size_t depth = 0) {
        budget.step();
        if (depth > 128) throw Limited{};
        if (rational_coefficient(expression)) {
            bound_coefficient(expression);
            return exact_zero(expression) ? Polynomial{} : Polynomial{{Monomial{}, expression}};
        }
        // Inexact and complex numeric coefficients do not belong to Q.
        if (se::is_a_Number(*expression)) throw Limited{};
        if (se::is_a<se::Add>(*expression)) {
            Polynomial result;
            for (const auto& argument : expression->get_args())
                for (const auto& [monomial, coefficient] : convert(argument, depth + 1))
                    add_term(result, monomial, coefficient, budget);
            return result;
        }
        if (se::is_a<se::Mul>(*expression)) {
            Polynomial result{{Monomial{}, se::one}};
            for (const auto& argument : expression->get_args())
                result = multiply_polynomials(result, convert(argument, depth + 1), budget);
            return result;
        }
        if (se::is_a<se::Pow>(*expression)) {
            const auto& power = se::down_cast<const se::Pow&>(*expression);
            if (se::is_a<se::Integer>(*power.get_exp())) {
                const auto& integer = se::down_cast<const se::Integer&>(*power.get_exp()).as_integer_class();
                if (integer >= 0) {
                    if (integer > degree_limit) throw Limited{};
                    auto exponent = static_cast<uint32_t>(se::down_cast<const se::Integer&>(*power.get_exp()).as_int());
                    // Match the outer normalizer's exact trigonometric identity.
                    // Functions with other arguments remain independent atoms.
                    if (se::is_a<se::Cos>(*power.get_base()) && exponent > 0 && exponent <= 12 && exponent % 2 == 0) {
                        const auto& cosine = se::down_cast<const se::Cos&>(*power.get_base());
                        auto square = power_polynomial(atom(se::sin(cosine.get_arg())), 2, budget);
                        Polynomial identity{{Monomial{}, se::one}};
                        for (const auto& [monomial, coefficient] : square)
                            add_term(identity, monomial, se::neg(coefficient), budget);
                        return power_polynomial(identity, exponent / 2, budget);
                    }
                    return power_polynomial(convert(power.get_base(), depth + 1), exponent, budget);
                }
            }
        }
        // A Symbol, Function, Derivative, or non-polynomial power is an opaque
        // algebraic atom. This can miss identities but cannot invent one.
        return atom(expression);
    }

    Expr reconstruct(const Polynomial& polynomial) {
        se::vec_basic terms;
        for (const auto& [monomial, coefficient] : polynomial) {
            budget.step();
            se::vec_basic factors{coefficient};
            for (const auto& [index, exponent] : monomial) {
                budget.step();
                factors.push_back(exponent == 1 ? atoms[index] : se::pow(atoms[index], se::integer(exponent)));
            }
            terms.push_back(se::mul(factors));
        }
        if (terms.empty()) return se::zero;
        return se::add(terms);
    }
};

bool exact_divide(const Polynomial& numerator, const Polynomial& divisor,
                  Polynomial& quotient, Budget& budget) {
    if (divisor.empty()) return false;
    auto remaining = numerator;
    quotient.clear();
    const auto& [divisor_monomial, divisor_coefficient] = *divisor.rbegin();
    while (!remaining.empty()) {
        budget.step();
        const auto leading = *remaining.rbegin();
        Monomial monomial;
        if (!divide_monomials(leading.first, divisor_monomial, monomial, budget)) return false;
        auto coefficient = se::div(leading.second, divisor_coefficient);
        bound_coefficient(coefficient);
        add_term(quotient, monomial, coefficient, budget);
        for (const auto& [dm, dc] : divisor) {
            auto product = multiply_monomials(monomial, dm, budget);
            budget.step();
            add_term(remaining, product, se::neg(se::mul(coefficient, dc)), budget);
        }
        // A proper graded lex division strictly decreases the leading monomial.
        // Retain this guard so an implementation change can only fail closed.
        if (!remaining.empty() && !GradedLex{}(remaining.rbegin()->first, leading.first)) throw Limited{};
    }
    return true;
}

struct DenominatorFactor { Expr base; uint32_t remaining = 1; };

void denominator_factors(const Expr& expression, std::vector<DenominatorFactor>& factors,
                         Expr& numeric, Budget& budget) {
    budget.step();
    if (rational_coefficient(expression)) {
        if (exact_zero(expression)) throw Limited{};
        numeric = se::mul(numeric, expression); bound_coefficient(numeric);
        return;
    }
    if (se::is_a_Number(*expression)) throw Limited{};
    if (se::is_a<se::Mul>(*expression)) {
        for (const auto& argument : expression->get_args()) denominator_factors(argument, factors, numeric, budget);
        return;
    }
    if (factors.size() >= atom_limit) throw Limited{};
    if (se::is_a<se::Pow>(*expression)) {
        const auto& power = se::down_cast<const se::Pow&>(*expression);
        if (se::is_a<se::Integer>(*power.get_exp())) {
            const auto& integer = se::down_cast<const se::Integer&>(*power.get_exp()).as_integer_class();
            if (integer > 0 && integer <= degree_limit) {
                auto exponent = static_cast<uint32_t>(se::down_cast<const se::Integer&>(*power.get_exp()).as_int());
                // Retain cos² as a known factor so its polynomial image is
                // 1-sin², matching an already normalized numerator.
                if (se::is_a<se::Cos>(*power.get_base()) && exponent <= 12 && exponent % 2 == 0)
                    factors.push_back({se::pow(power.get_base(), se::integer(2)), exponent / 2});
                else factors.push_back({power.get_base(), exponent});
                return;
            }
        }
    }
    factors.push_back({expression, 1});
}

} // namespace

Expr cancel_denominator_factors(const Expr& expanded_numerator, const Expr& factored_denominator) {
    check_cancelled();
    // Rational scalar denominators already cancel through SymEngine arithmetic.
    if (rational_coefficient(factored_denominator)) return se::div(expanded_numerator, factored_denominator);
    try {
        Budget budget;
        Algebra algebra(budget);
        auto numerator = algebra.convert(expanded_numerator);
        std::vector<DenominatorFactor> factors;
        Expr numeric = se::one;
        denominator_factors(factored_denominator, factors, numeric, budget);
        bool improved = false;
        for (auto& factor : factors) {
            auto divisor = algebra.convert(factor.base);
            if (divisor.empty()) throw Limited{};
            while (factor.remaining) {
                Polynomial quotient;
                if (!exact_divide(numerator, divisor, quotient, budget)) break;
                numerator = std::move(quotient);
                --factor.remaining;
                improved = true;
            }
        }
        if (improved) {
            auto reduced_numerator = algebra.reconstruct(numerator);
            se::vec_basic remaining{numeric};
            for (const auto& factor : factors) if (factor.remaining) {
                budget.step();
                remaining.push_back(factor.remaining == 1 ? factor.base : se::pow(factor.base, se::integer(factor.remaining)));
            }
            return se::div(reduced_numerator, se::mul(remaining));
        }
    } catch (const Limited&) {
        // No partial result escapes a budget failure.
    }
    check_cancelled();
    return se::div(expanded_numerator, factored_denominator);
}

} // namespace tensor
