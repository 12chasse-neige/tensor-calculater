#include "tensor/core.hpp"
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/integer.h>
#include <symengine/number.h>
#include <symengine/functions.h>
#include <symengine/matrix.h>
#include <chrono>
#include <stdexcept>

namespace tensor {
namespace se = SymEngine;
namespace {
constexpr const char* convention =
    "R^rho_{sigma mu nu} = d_nu Gamma^rho_{mu sigma} - d_mu Gamma^rho_{nu sigma} "
    "+ Gamma^rho_{nu alpha} Gamma^alpha_{mu sigma} - Gamma^rho_{mu alpha} Gamma^alpha_{nu sigma}; "
    "Ricci_{sigma nu} = R^rho_{sigma rho nu}. Indices are zero-based.";

Expr component(const Components& tensor, const Index& index) {
    auto it = tensor.find(index);
    if (it == tensor.end()) return se::zero;
    return it->second;
}

JSON printed(const Expr& value) {
    return {{"expression", value->__str__()}, {"latex", expression_latex(value)},
            {"zero_status", exact_zero(value) ? "zero" : se::is_a_Number(*value) ? "nonzero" : "undetermined"}};
}

JSON tensor_result(const std::string& key, const std::string& name,
                   const std::string& symbol, const std::string& variance,
                   int n, const Components& tensor) {
    JSON list = JSON::array();
    for (const auto& [index, value] : tensor) {
        auto entry = printed(value); entry["indices"] = index; list.push_back(std::move(entry));
    }
    return {{"key",key},{"name",name},{"symbol",symbol},{"variance",variance},
            {"rank",variance.size()},{"shape",std::vector<int>(variance.size(),n)}, {"components",list}};
}

class Calculator {
    const Input& input;
    const Progress& progress;
    int n;
    std::vector<Expr> inverse;
    Components gamma, riemann, ricci;
    std::map<Index, Expr> metric_derivatives;
    struct DerivativeKeyLess {
        bool operator()(const std::pair<Expr,int>& a, const std::pair<Expr,int>& b) const {
            int comparison = a.first->__cmp__(*b.first);
            return comparison < 0 || (comparison == 0 && a.second < b.second);
        }
    };
    std::map<std::pair<Expr,int>, Expr, DerivativeKeyLess> expression_derivatives;
    std::vector<std::vector<std::pair<int,Expr>>> inverse_rows, metric_rows;

    Expr metric(int a, int b) const { return input.metric[a*n+b]; }
    Expr inv(int a, int b) const { return inverse[a*n+b]; }
    void store(Components& tensor, const Index& index, const Expr& value) {
        check_cancelled(); auto simplified = normalize(value);
        if (!exact_zero(simplified)) tensor[index] = simplified;
    }
    Expr metric_derivative(int a, int b, int c) {
        Index key{a,b,c};
        auto it = metric_derivatives.find(key);
        if (it != metric_derivatives.end()) return it->second;
        auto value = normalize(metric(a,b)->diff(input.coordinates[c]));
        metric_derivatives[key] = value; return value;
    }
    Expr derivative(const Expr& expression, int coordinate) {
        if (exact_zero(expression)) return se::zero;
        auto key = std::make_pair(expression,coordinate);
        auto it = expression_derivatives.find(key);
        if (it != expression_derivatives.end()) return it->second;
        check_cancelled();
        auto value = normalize(expression->diff(input.coordinates[coordinate]));
        expression_derivatives[key] = value; return value;
    }
    void compute_inverse() {
        progress("inverse_metric","Constructing the inverse metric…");
        bool diagonal = true;
        for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b)
            if (a != b && !exact_zero(metric(a,b))) diagonal = false;
        inverse.assign(n*n,se::zero);
        if (diagonal) {
            for (int a = 0; a < n; ++a) {
                auto value = normalize(metric(a,a));
                if (exact_zero(value)) throw std::runtime_error("Metric is singular and cannot be inverted.");
                inverse[a*n+a] = normalize(se::div(se::one,value));
            }
        } else {
            se::DenseMatrix matrix(n,n,input.metric), result(n,n);
            auto determinant = normalize(matrix.det());
            if (exact_zero(determinant)) throw std::runtime_error("Metric is singular and cannot be inverted.");
            check_cancelled();
            matrix.inv(result);
            for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) inverse[a*n+b] = normalize(result.get(a,b));
        }
        inverse_rows.resize(n); metric_rows.resize(n);
        for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) {
            if (!exact_zero(inv(a,b))) inverse_rows[a].emplace_back(b,inv(a,b));
            if (!exact_zero(metric(a,b))) metric_rows[a].emplace_back(b,metric(a,b));
        }
    }
    void compute_gamma() {
        progress("christoffel","Calculating Christoffel symbols…");
        // Gamma^rho_{mu nu} = (1/2) g^{rho sigma}
        // (d_mu g_{sigma nu} + d_nu g_{sigma mu} - d_sigma g_{mu nu}).
        // The Levi-Civita lower-index symmetry saves half of the derivatives.
        for (int rho = 0; rho < n; ++rho) for (int mu = 0; mu < n; ++mu) for (int nu = mu; nu < n; ++nu) {
            check_cancelled(); Expr total = se::zero;
            for (const auto& [sigma, g] : inverse_rows[rho]) {
                auto inner = se::sub(se::add(metric_derivative(sigma,nu,mu),metric_derivative(sigma,mu,nu)),metric_derivative(mu,nu,sigma));
                if (!exact_zero(inner)) total = se::add(total,se::mul(g,inner));
            }
            Index key{rho,mu,nu}; store(gamma,key,se::div(total,se::integer(2)));
            if (gamma.count(key) && mu != nu) gamma[{rho,nu,mu}] = gamma[key];
        }
    }
    Expr G(int a, int b, int c) const { return component(gamma,{a,b,c}); }
    void compute_riemann() {
        progress("riemann","Calculating Riemann curvature…");
        // Preserve the source calculator's sign: derivative nu precedes mu.
        // Only mu < nu is evaluated; the last pair is antisymmetric.
        for (int rho = 0; rho < n; ++rho) for (int sigma = 0; sigma < n; ++sigma)
            for (int mu = 0; mu < n; ++mu) for (int nu = mu+1; nu < n; ++nu) {
                check_cancelled();
                auto total = se::sub(derivative(G(rho,mu,sigma),nu),derivative(G(rho,nu,sigma),mu));
                for (int alpha = 0; alpha < n; ++alpha)
                    total = se::add(total,se::sub(se::mul(G(rho,nu,alpha),G(alpha,mu,sigma)),se::mul(G(rho,mu,alpha),G(alpha,nu,sigma))));
                Index key{rho,sigma,mu,nu}; store(riemann,key,total);
                if (riemann.count(key)) riemann[{rho,sigma,nu,mu}] = se::neg(riemann[key]);
            }
    }
    void compute_ricci(bool have_riemann) {
        progress("ricci","Calculating the Ricci tensor…");
        // R_{sigma nu} = sum_rho R^rho_{sigma rho nu}. When the complete
        // Riemann tensor was not requested, evaluate this contraction directly.
        for (int sigma = 0; sigma < n; ++sigma) for (int nu = sigma; nu < n; ++nu) {
            check_cancelled(); Expr total = se::zero;
            for (int rho = 0; rho < n; ++rho) {
                if (have_riemann) total = se::add(total,component(riemann,{rho,sigma,rho,nu}));
                else {
                    total = se::add(total,se::sub(derivative(G(rho,rho,sigma),nu),derivative(G(rho,nu,sigma),rho)));
                    for (int alpha = 0; alpha < n; ++alpha)
                        total = se::add(total,se::sub(se::mul(G(rho,nu,alpha),G(alpha,rho,sigma)),se::mul(G(rho,rho,alpha),G(alpha,nu,sigma))));
                }
            }
            Index key{sigma,nu}; store(ricci,key,total);
            if (ricci.count(key) && sigma != nu) ricci[{nu,sigma}] = ricci[key];
        }
    }
    Expr scalar_curvature() {
        progress("ricci_scalar","Contracting the Ricci scalar…");
        Expr total = se::zero;
        for (int a = 0; a < n; ++a) for (const auto& [b,g] : inverse_rows[a]) total = se::add(total,se::mul(g,component(ricci,{a,b})));
        return normalize(total);
    }
    Expr kerr_kretschmann() {
        if (input.names != std::vector<std::string>{"t","r","theta","phi"} || !input.symbols.count("M") || !input.symbols.count("a")) return {};
        auto r = input.symbols.at("r"), theta = input.symbols.at("theta"), M = input.symbols.at("M"), a = input.symbols.at("a");
        Input check = input;
        const std::vector<std::string> entries = {
            "-(1-2*M*r/(r^2+a^2*cos(theta)^2))", "0", "0", "-2*M*a*r*sin(theta)^2/(r^2+a^2*cos(theta)^2)",
            "0", "(r^2+a^2*cos(theta)^2)/(r^2-2*M*r+a^2)", "0", "0",
            "0", "0", "r^2+a^2*cos(theta)^2", "0",
            "-2*M*a*r*sin(theta)^2/(r^2+a^2*cos(theta)^2)", "0", "0", "(r^2+a^2+2*M*a^2*r*sin(theta)^2/(r^2+a^2*cos(theta)^2))*sin(theta)^2"
        };
        bool original_sign = true, reversed_sign = true;
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto expected = parse_expression(entries[i],check);
            original_sign = original_sign && exact_zero(normalize(se::sub(input.metric[i],expected)));
            reversed_sign = reversed_sign && exact_zero(normalize(se::add(input.metric[i],expected)));
            if (!original_sign && !reversed_sign) return {};
        }
        auto c = se::cos(theta);
        auto p = [](const Expr& x, int exponent) { return se::pow(x,se::integer(exponent)); };
        auto numerator = se::sub(se::add(se::sub(p(r,6),se::mul(se::integer(15),se::mul(p(a,2),se::mul(p(r,4),p(c,2))))),se::mul(se::integer(15),se::mul(p(a,4),se::mul(p(r,2),p(c,4))))),se::mul(p(a,6),p(c,6)));
        auto sigma = se::add(p(r,2),se::mul(p(a,2),p(c,2)));
        return se::div(se::mul(se::integer(48),se::mul(p(M,2),numerator)),p(sigma,6));
    }
    Expr kretschmann() {
        progress("kretschmann","Contracting the Kretschmann scalar…");
        Components lowered;
        // g_{a rho} R^rho_{b c d}. Symmetry of g allows access by row rho.
        for (const auto& [index,value] : riemann)
            for (const auto& [a,g] : metric_rows[index[0]]) {
                Index key{a,index[1],index[2],index[3]};
                lowered[key] = se::add(component(lowered,key),se::mul(g,value));
            }
        for (auto it = lowered.begin(); it != lowered.end();) {
            auto value = normalize(it->second);
            if (exact_zero(value)) it = lowered.erase(it);
            else { it->second = value; ++it; }
        }
        Expr total = se::zero;
        for (const auto& [index,value] : lowered) {
            check_cancelled(); Expr raised = se::zero;
            for (const auto& [e,gae] : inverse_rows[index[0]])
                for (const auto& [f,gbf] : inverse_rows[index[1]])
                    for (const auto& [g,gcg] : inverse_rows[index[2]])
                        for (const auto& [h,gdh] : inverse_rows[index[3]]) {
                            auto other = component(lowered,{e,f,g,h});
                            if (!exact_zero(other)) raised = se::add(raised,se::mul(se::mul(gae,gbf),se::mul(se::mul(gcg,gdh),other)));
                        }
            total = se::add(total,se::mul(value,normalize(raised)));
        }
        return normalize(total);
    }
public:
    Calculator(const Input& i, const Progress& p) : input(i), progress(p), n(static_cast<int>(i.names.size())) {}
    JSON run() {
        auto start = std::chrono::steady_clock::now();
        JSON tensors = JSON::array(), scalars = JSON::array(), warnings = JSON::array();
        auto selected = [&](const std::string& key) { return input.outputs.count(key) != 0; };
        compute_inverse();
        Expr known_k;
        if (selected("kretschmann")) known_k = kerr_kretschmann();
        bool need_riemann = selected("riemann") || (selected("kretschmann") && known_k.is_null());
        bool need_ricci = selected("ricci") || selected("ricci_scalar");
        if (need_riemann || need_ricci || selected("christoffel")) compute_gamma();
        if (need_riemann) compute_riemann();
        if (need_ricci) compute_ricci(need_riemann);
        // Apply the convention once, after either Ricci calculation path.
        // Quadratic invariants are unchanged; the scalar uses signed Ricci.
        if (input.riemann_sign == -1) {
            for (auto& [index, value] : riemann) value = se::neg(value);
            for (auto& [index, value] : ricci) value = se::neg(value);
        }
        if (selected("inverse_metric")) {
            Components components;
            for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b)
                if (!exact_zero(inv(a,b))) components[{a,b}] = inv(a,b);
            tensors.push_back(tensor_result("inverse_metric","Inverse metric","g","uu",n,components));
        }
        if (selected("christoffel")) tensors.push_back(tensor_result("christoffel","Christoffel symbols","\\Gamma","ull",n,gamma));
        if (selected("riemann")) tensors.push_back(tensor_result("riemann","Riemann tensor","R","ulll",n,riemann));
        if (selected("ricci")) tensors.push_back(tensor_result("ricci","Ricci tensor","R","ll",n,ricci));
        if (selected("ricci_scalar")) {
            auto item = printed(scalar_curvature()); item["key"] = "ricci_scalar"; item["name"] = "Ricci scalar"; item["symbol"] = "R"; scalars.push_back(item);
        }
        if (selected("kretschmann")) {
            if (!known_k.is_null()) {
                progress("kretschmann","Using the verified Kerr closed-form invariant…");
                warnings.push_back("Kerr Kretschmann scalar uses a recognized closed form; other selected tensors are calculated from the metric.");
            }
            auto item = printed(!known_k.is_null() ? known_k : kretschmann()); item["key"] = "kretschmann"; item["name"] = "Kretschmann scalar"; item["symbol"] = "K"; scalars.push_back(item);
        }
        warnings.push_back("Results apply where the metric is invertible and the expressions are defined. Only proven symbolic zeros are omitted; unsimplified components may still vanish identically.");
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return {{"coordinates",input.names},{"elapsed_seconds",elapsed},{"convention",input.riemann_sign == 1 ? std::string(convention) : std::string("R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu alpha} Gamma^alpha_{nu sigma} - Gamma^rho_{nu alpha} Gamma^alpha_{mu sigma}; Ricci_{sigma nu} = R^rho_{sigma rho nu}. Indices are zero-based.")},{"tensors",tensors},{"scalars",scalars},{"warnings",warnings}};
    }
};
} // namespace

JSON calculate(const Input& input, const Progress& progress) { return Calculator(input,progress).run(); }
} // namespace tensor
