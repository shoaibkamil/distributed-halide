#include <algorithm>

#include <nfm_solver.h>

#include "NfmToHalide.h"

#include "HalideNfmConverter.h"
#include "IREquality.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "Module.h"
#include "Schedule.h"
#include "Simplify.h"

namespace Halide {
namespace Internal {

using namespace Nfm;
using namespace Nfm::Internal;

using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace {

// Iterate over values from last_checked-1 to beginning. If the element is in vector v,
// return that index. Otherwise, return -1
int find_first_index_in_vec(const vector<string> *v, const vector<Expr>& values, int last_checked) {
    assert(last_checked >= 0 && last_checked <= (int)values.size());
    if ((v != NULL) && (values.size() > 0)) {
        for (int i = last_checked-1; i >= 0; --i) {
            const Variable *var = values[i].as<Variable>();
            assert(var != NULL);
            auto iter = std::find(v->begin(), v->end(), var->name);
            if (iter != v->end()) {
                return i;
            }
        }
    }
    return -1;
}

bool is_val_in_vector(const vector<string> *v, const string& val) {
    if (v != NULL) {
        auto iter = std::find(v->begin(), v->end(), val);
        if (iter != v->end()) {
            return true;
        }
    }
    return false;
}

enum BoundType {
    UNDEFINED,
    EQUAL,
    LOWER_BOUND,
    UPPER_BOUND,
    CONDITION
};

enum LowerUpperBoundType {
    BOUND_UNDEFINED,
    BOUND_EQUALITY,
    BOUND_INEQUALITY
};

struct BoundExpr {
    BoundType type;
    Expr value;

    explicit BoundExpr() : type(UNDEFINED) {}
    BoundExpr(const Expr& val, BoundType type)
        : type(type), value(val) {}

    bool is_defined() const { return (type != UNDEFINED); }
    void simplify() {
        if (value.defined()) {
            value = Halide::Internal::simplify(value);
        }
    }
};

// If condition is defined, it means that it's always true.
// If lb or ub is undefined, it means that it is unbounded, i.e, -infinity <= x
struct LowerUpperBound {
    string var;
    LowerUpperBoundType type;
    // No condition means it's always true
    Expr condition; // OR of AND
    Expr lb; // Lower bound: lb <= var
    Expr ub; // Upper bound: var <= ub

    explicit LowerUpperBound(const string& var) : var(var), type(BOUND_UNDEFINED) {}
    LowerUpperBound(const string& var, const Expr& cond,
                    const Expr& lb, const Expr& ub,
                    LowerUpperBoundType type)
        : var(var), type(type), condition(cond), lb(lb), ub(ub) {}

    LowerUpperBound(const string& var, const Expr& cond,
                    const Expr& lb,
                    const Expr& ub)
        : LowerUpperBound(var, cond, lb, ub, BOUND_INEQUALITY) {}

    bool has_condition() { return condition.defined(); }
    bool has_lower_bound() { return lb.defined(); }
    bool has_upper_bound() { return ub.defined(); }
    bool is_equality() const { return (type == BOUND_EQUALITY); }
    bool is_defined() const { return (type != BOUND_UNDEFINED); }

    // Return true if the condition is always false
    bool is_always_false() const {
        if (condition.defined() && is_zero(condition)) {
            return true;
        }
        return false;
    }

    void simplify() {
        if (condition.defined()) {
            condition = Halide::Internal::simplify(condition);
        }
        if (lb.defined()) {
            lb = Halide::Internal::simplify(lb);
        }
        if (ub.defined()) {
            ub = Halide::Internal::simplify(ub);
        }
    }
};

template <typename T>
vector<T> operator+(const vector<T>& a, const vector<T>& b) {
    assert(a.size() == b.size());

    vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}

template <typename T>
vector<T> operator-(const vector<T>& a, const vector<T>& b) {
    assert(a.size() == b.size());

    vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::minus<T>());
    return result;
}

Expr exponent_to_expr(const vector<int>& p_exp, const vector<Expr>& var) {
    user_assert(p_exp.size() > 0) << "p_exp size should be bigger than 0";
    assert(p_exp.size() == var.size());
    Expr expr = 1;
    for (size_t i = 0; i < p_exp.size(); ++i) {
        assert(p_exp[i] >= 0);
        if (p_exp[i] >= 1) {
            for (size_t j = p_exp[i]; j > 0; --j) {
                expr = Mul::make(expr, var[i]);
            }
        }
    }
    expr = simplify(expr);
    return expr;
}

int find_dim_index(const vector<string>& dim_vars, const string& dim_name) {
    std::string var = dim_name;
    /*if (ends_with(dim_name, ".base")) {
        var = dim_name.substr(0, dim_name.size()-5);
    }*/
    for (size_t i = 0; i < dim_vars.size(); ++i) {
        if (ends_with(var, dim_vars[i])) {
            return (int)i;
        }
    }
    return -1;
}

// Sort in the ascending order of the sym_const_vars followed by dim_vars.
bool compare_dims(const vector<string> sym_const_vars, const vector<string>& dim_vars,
                  const string& dim1, const string& dim2) {
    int idx_dim1 = find_dim_index(sym_const_vars, dim1);
    if (idx_dim1 == -1) {
        idx_dim1 = find_dim_index(dim_vars, dim1);
        assert(idx_dim1 != -1);
        idx_dim1 += sym_const_vars.size();
    }
    int idx_dim2 = find_dim_index(sym_const_vars, dim2);
    if (idx_dim2 == -1) {
        idx_dim2 = find_dim_index(dim_vars, dim2);
        assert(idx_dim2 != -1);
        idx_dim2 += sym_const_vars.size();
    }
    return idx_dim1 < idx_dim2;
}

Expr convert_to_let_helper(const vector<Expr>& ands, size_t start, const Expr& val) {
    assert(ands.size() > 0);
    assert(ands.size()-1 >= start);

    Expr expr;
    {
        const EQ *cond_eq = ands[ands.size()-1].as<EQ>();
        assert(cond_eq != NULL);
        Expr value = simplify(val);
        const Variable *var = cond_eq->a.as<Variable>();
        expr = Let::make(var->name, cond_eq->b, value);
    }
    for (int i = ands.size()-2; i >= (int)start; --i) {
        const EQ *cond_eq = ands[i].as<EQ>();
        assert(cond_eq != NULL);
        const Variable *var = cond_eq->a.as<Variable>();
        expr = Let::make(var->name, cond_eq->b, expr);
    }
    return expr;
}

Expr convert_nfm_poly_coeff_to_halide_expr(Type type, const NfmPolyCoeff& poly,
                                           const vector<Expr>& sym_const_vars) {
    //std::cout << "convert_nfm_poly_coeff_to_halide_expr " << poly << "\n";
    Expr result = 0;
    for (auto iter : poly.get_terms()) {
        if (iter.second == 0) {
            continue;
        } else if (iter.second == 1) {
            result = Add::make(result, exponent_to_expr(iter.first, sym_const_vars));
        } else {
            result = Add::make(result,
                Mul::make(iter.second, exponent_to_expr(iter.first, sym_const_vars)));
        }

    }
    //std::cout << "convert_nfm_poly_coeff_to_halide_expr result: " << result << "\n";
    return simplify(result);
}

Expr convert_nfm_poly_to_halide_expr(Type type, const NfmPoly& poly,
                                     const vector<Expr>& sym_const_vars,
                                     const vector<Expr>& dim_vars) {
    //std::cout << "convert_nfm_poly_to_halide_expr " << poly << "\n";
    Expr result = 0;
    for (auto iter : poly.get_terms()) {
        if (iter.second.is_zero()) {
            continue;
        } else if (iter.second.is_one()) {
            result = Add::make(result, exponent_to_expr(iter.first, dim_vars));
        } else {
             Expr coeff_expr = convert_nfm_poly_coeff_to_halide_expr(
                type, iter.second, sym_const_vars);
            result = Add::make(result, Mul::make(coeff_expr, exponent_to_expr(iter.first, dim_vars)));
        }
    }
    //std::cout << "convert_nfm_poly_to_halide_expr result: " << result << "\n";
    return simplify(result);
}

Expr convert_nfm_context_poly_coeff_to_halide_expr(
        Type type, const NfmPolyCoeff& poly, const vector<Expr>& sym_const_vars,
        bool is_equality, const vector<string> *let_assignments, bool *is_let=NULL) {
    //std::cout << "convert_nfm_poly_coeff_to_halide_expr " << poly << " to halide\n";
    if (is_let != NULL) {
        *is_let = false;
    }
    if (!is_equality) { // Inequality can't be an let assignment; no need to check
        //std::cout << "Converting context linear cst: " << poly << " >= 0\n";
        Expr expr = convert_nfm_poly_coeff_to_halide_expr(type, poly, sym_const_vars);
        expr = GE::make(simplify(expr), 0);
        return expr;
    }

    //std::cout << "Converting context linear cst: " << poly << " == 0\n";

    int idx = find_first_index_in_vec(let_assignments, sym_const_vars, sym_const_vars.size());
    if (idx == -1) {
        Expr expr = convert_nfm_poly_coeff_to_halide_expr(type, poly, sym_const_vars);
        expr = EQ::make(simplify(expr), 0);
        return expr;
    }

    Expr lhs;
    Expr rhs;
    for (; idx >= 0; idx = find_first_index_in_vec(let_assignments, sym_const_vars, idx)) {
        std::pair<NfmPolyCoeff, NfmPolyCoeff> term = poly.get_coeff_involving_dim(idx);
        const auto& coeff = term.second;
        if (!coeff.is_zero()) {
            lhs = convert_nfm_poly_coeff_to_halide_expr(type, term.first.exquo(term.second), sym_const_vars);

            NfmPolyCoeff rhs_poly = poly-term.first;
            if (coeff.is_one()) {
                rhs = convert_nfm_poly_coeff_to_halide_expr(type, rhs_poly.neg(), sym_const_vars);
            } else if (coeff.is_neg_one()) {
                rhs = convert_nfm_poly_coeff_to_halide_expr(type, rhs_poly, sym_const_vars);
            } else {
                rhs = convert_nfm_poly_coeff_to_halide_expr(type, rhs_poly.neg(), sym_const_vars);
                Expr denom = convert_nfm_poly_coeff_to_halide_expr(type, coeff, sym_const_vars);
                rhs = cast(type, ceil(rhs/denom));
            }
            if (is_let != NULL) {
                *is_let = true;
            }
            break;
        }
    }
    assert(idx >= 0);
    assert(lhs.defined());
    assert(rhs.defined());
    lhs = simplify(lhs);
    rhs = simplify(rhs);
    Expr expr = EQ::make(lhs, rhs);
    //std::cout << "convert_nfm_context_poly_coeff_to_halide_expr result: " << expr << "\n\n";
    return expr;
}

Expr convert_nfm_context_domain_to_halide_expr(Type type, NfmContextDomain& ctx_dom,
                                               const vector<Expr>& sym_const_vars,
                                               const vector<string> *let_assignments) {
    Expr temp;
    ctx_dom.simplify();
    if (ctx_dom.is_empty() || ctx_dom.is_universe()) {
        return temp;
    }
    const vector<NfmContext>& linear = ctx_dom.get_linear_contexts();
    const vector<NfmContext>& non_linear = ctx_dom.get_non_linear_contexts();

    for (const auto& ctx : linear) {
        Expr ctx_expr = convert_nfm_context_poly_coeff_to_halide_expr(
            type, ctx.get_context(), sym_const_vars, ctx.is_equality(),
            let_assignments);
        if (temp.defined()) {
            temp = And::make(temp, ctx_expr);
        } else {
            temp = ctx_expr;
        }
    }
    for (const auto& ctx : non_linear) {
        Expr ctx_expr = convert_nfm_context_poly_coeff_to_halide_expr(
            type, ctx.get_context(), sym_const_vars, ctx.is_equality(),
            let_assignments);
        if (temp.defined()) {
            temp = And::make(temp, ctx_expr);
        } else {
            temp = ctx_expr;
        }
    }
    return temp;
}

std::pair<set<Expr, IRDeepCompare>, set<Expr, IRDeepCompare>>
convert_nfm_context_domain_to_halide_expr_pair(Type type, NfmContextDomain& ctx_dom,
                                               const vector<Expr>& sym_const_vars,
                                               const vector<string> *let_assignments) {
    // lets, conditional pairs
    std::pair<set<Expr, IRDeepCompare>, set<Expr, IRDeepCompare>> result;
    ctx_dom.simplify();
    if (ctx_dom.is_empty() || ctx_dom.is_universe()) {
        return result;
    }
    const vector<NfmContext>& linear = ctx_dom.get_linear_contexts();
    const vector<NfmContext>& non_linear = ctx_dom.get_non_linear_contexts();

    for (const auto& ctx : linear) {
        bool is_let = false;
        Expr ctx_expr = convert_nfm_context_poly_coeff_to_halide_expr(
            type, ctx.get_context(), sym_const_vars, ctx.is_equality(),
            let_assignments, &is_let);
        if (is_let) {
            result.first.insert(ctx_expr);
        } else {
            result.second.insert(ctx_expr);
        }
    }
    for (const auto& ctx : non_linear) {
        bool is_let = false;
        Expr ctx_expr = convert_nfm_context_poly_coeff_to_halide_expr(
            type, ctx.get_context(), sym_const_vars, ctx.is_equality(),
            let_assignments, &is_let);
        if (is_let) {
            result.first.insert(ctx_expr);
        } else {
            result.second.insert(ctx_expr);
        }
    }
    return result;
}

Expr convert_nfm_constraint_to_halide_expr(
        Type type, const NfmContextDomain& ctx_dom, const NfmConstraint& cst,
        const vector<Expr>& sym_const_vars, const vector<Expr>& dim_vars,
        const vector<string> *let_assignments, bool *is_let=NULL) {
    //std::cout << "convert_nfm_constraint_to_halide_expr " << cst << "\n";
    if (is_let != NULL) {
        *is_let = false;
    }
    Expr result;
    if (!cst.is_equality()) {
        result = convert_nfm_poly_to_halide_expr(type, cst.get_constraint(),
                                                 sym_const_vars, dim_vars);
        result = GE::make(result, 0);
        return result;
    }

    int idx = find_first_index_in_vec(let_assignments, dim_vars, dim_vars.size());
    if (idx == -1) {
        result = convert_nfm_poly_to_halide_expr(type, cst.get_constraint(),
                                                 sym_const_vars, dim_vars);
        result = EQ::make(result, 0);
        return result;
    }

    for (; idx >= 0; idx = find_first_index_in_vec(let_assignments, dim_vars, idx)) {
        NfmPolyCoeff coeff = cst.get_coeff(idx);
        if (!NfmSolver::nfm_poly_coeff_is_zero(ctx_dom, coeff)) {
            NfmPoly poly = cst.get_constraint().drop_term(idx);
            if (coeff.is_one()) {
                Expr temp = convert_nfm_poly_to_halide_expr(type, poly.neg(), sym_const_vars, dim_vars);
                result = EQ::make(dim_vars[idx], temp);
            } else if (coeff.is_neg_one()) {
                Expr temp = convert_nfm_poly_to_halide_expr(type, poly, sym_const_vars, dim_vars);
                result = EQ::make(dim_vars[idx], temp);
            } else {
                Expr temp = convert_nfm_poly_to_halide_expr(type, poly.neg(), sym_const_vars, dim_vars);
                Expr denom = convert_nfm_poly_coeff_to_halide_expr(
                    type, coeff, sym_const_vars);
                result = EQ::make(dim_vars[idx], cast(type, ceil(simplify(temp/denom))));
            }
            if (is_let != NULL) {
                *is_let = true;
            }
            break;
        }
    }
    assert(idx >= 0);

    //std::cout << "convert_nfm_constraint_to_halide_expr result: " << result << "\n";
    return simplify(result);
}

Expr convert_nfm_domain_to_halide_expr(
        Type type, NfmDomain& dom, const vector<Expr>& sym_const_vars,
        const vector<Expr>& dim_vars,
        const vector<string> *let_assignments) {
    //std::cout << "Converting domain " << dom << " to halide\n";

    NfmContextDomain& ctx_dom = dom.get_context_domain();
    auto ctx_let_cond = convert_nfm_context_domain_to_halide_expr_pair(
        type, ctx_dom, sym_const_vars, let_assignments);

    set<Expr, IRDeepCompare> let_exprs;
    set<Expr, IRDeepCompare> cond_exprs;
    for (auto& expr : ctx_let_cond.first) {
        if (expr.defined() && is_zero(expr)) { // The condition is never true
            return Expr();
        }
        let_exprs.insert(expr);
    }
    for (auto& expr : ctx_let_cond.second) {
        if (expr.defined() && is_zero(expr)) { // The condition is never true
            return Expr();
        }
        cond_exprs.insert(expr);
    }

    const auto& eq_constraints = dom.get_equalities();
    for (size_t i = 0; i < eq_constraints.size(); ++i) {
        bool is_let = false;
        Expr expr = convert_nfm_constraint_to_halide_expr(
            type, ctx_dom, eq_constraints[i], sym_const_vars, dim_vars,
            let_assignments, &is_let);
        if (is_let) {
            let_exprs.insert(expr);
        } else {
            cond_exprs.insert(expr);
        }
    }

    const auto& ineqs_constraints = dom.get_inequalities();
    for (size_t i = 0; i < ineqs_constraints.size(); ++i) {
        bool is_let = false;
        Expr expr = convert_nfm_constraint_to_halide_expr(
            type, ctx_dom, ineqs_constraints[i], sym_const_vars, dim_vars,
            let_assignments, &is_let);
        if (is_let) {
            let_exprs.insert(expr);
        } else {
            cond_exprs.insert(expr);
        }
    }

    Expr and_expr;
    for (auto& expr : cond_exprs) {
        if (and_expr.defined()) {
            and_expr = And::make(and_expr, expr);
        } else {
            and_expr = expr;
        }
    }
    if (let_exprs.size() > 0) {
        and_expr = convert_to_let_helper(
            vector<Expr>(let_exprs.begin(), let_exprs.end()), 0, and_expr);
    }
    and_expr = simplify(and_expr);
    return and_expr;
}

// Only process conditional constraint
BoundExpr convert_nfm_constraint_to_bound_expr_condition(
        Type type, const NfmContextDomain& ctx_dom, NfmConstraint& cst,
        int start_dim_idx, int end_dim_idx, const vector<Expr>& sym_const_vars,
        const vector<Expr>& dim_vars, const vector<string> *let_assignments) {
    assert(start_dim_idx <= end_dim_idx);
    assert(start_dim_idx >= 0 && end_dim_idx >= 0);

    //std::cout << "Converting condition cst " << cst << " to halide\n";
    BoundExpr result;
    result.type = CONDITION;
    result.value = convert_nfm_constraint_to_halide_expr(
        type, ctx_dom, cst, sym_const_vars, dim_vars, let_assignments);
    //std::cout << "  CONDITION result: " << result.value << "\n";
    return result;
}

// Only process conditional constraint
BoundExpr convert_nfm_constraint_to_bound_expr_condition(
        Type type, const NfmContextDomain& ctx_dom, NfmConstraint& cst,
        int dim_idx, const vector<Expr>& sym_const_vars,
        const vector<Expr>& dim_vars, const vector<string> *let_assignments) {
    return convert_nfm_constraint_to_bound_expr_condition(type, ctx_dom, cst,
        dim_idx, dim_idx, sym_const_vars, dim_vars, let_assignments);
}

// Ignore constraint not involving the dim (return undefined BoundExpr)
BoundExpr convert_nfm_constraint_to_bound_expr_non_condition(
        Type type, const NfmContextDomain& ctx_dom, NfmConstraint& cst,
        int dim_idx, const vector<Expr>& sym_const_vars,
        const vector<Expr>& dim_vars) {
    user_assert(dim_idx >= 0 && dim_idx < (int)dim_vars.size())
        << "dim_idx: " << dim_idx << "; size: " << dim_vars.size() << "\n";
    assert(NfmSolver::nfm_constraint_involves(ctx_dom, cst, dim_idx));

    //std::cout << "Converting cst " << cst << " to halide\n";
    BoundExpr result;
    NfmPolyCoeff coeff = cst.get_coeff(dim_idx);
    NfmPoly poly = cst.get_constraint().drop_term(dim_idx);
    if (NfmSolver::nfm_poly_coeff_is_pos(ctx_dom, coeff)) {
        result.value = convert_nfm_poly_to_halide_expr(type, poly.neg(), sym_const_vars, dim_vars);
    } else {
        assert(NfmSolver::nfm_poly_coeff_is_neg(ctx_dom, coeff)); // Should have had been negative
        coeff = coeff.neg();
        result.value = convert_nfm_poly_to_halide_expr(type, poly, sym_const_vars, dim_vars);
    }
    if (cst.is_equality()) { // constant*var + ... = 0
        result.type = EQUAL;
        if (!coeff.is_one()) {
            Expr denom = convert_nfm_poly_coeff_to_halide_expr(
                type, coeff, sym_const_vars);
            result.value = cast(type, ceil(result.value/denom));
        }
        //std::cout << "  EQUAL result: " << dim_name << " == " << result.value << "\n";
    } else if (NfmSolver::nfm_constraint_is_lower_bound(ctx_dom, cst, dim_idx)) { // var + ... >= 0
        result.type = LOWER_BOUND;
        if (!coeff.is_one()) { // e.g 3*x - 13 >= 0
            assert(NfmSolver::nfm_poly_coeff_is_pos(ctx_dom, coeff));
            Expr denom = convert_nfm_poly_coeff_to_halide_expr(
                type, coeff, sym_const_vars);
            result.value = cast(type, ceil(result.value/denom));
        }
        //std::cout << "  LOWER_BOUND result: " << dim_name << " >= " << result.value << "\n";
    } else if (NfmSolver::nfm_constraint_is_upper_bound(ctx_dom, cst, dim_idx)) { // -var + ...  >= 0
        result.type = UPPER_BOUND;
        if (!coeff.is_one()) { // e.g -3*x - 13 >= 0
            assert(NfmSolver::nfm_poly_coeff_is_pos(ctx_dom, coeff));
            Expr denom = convert_nfm_poly_coeff_to_halide_expr(
                type, coeff, sym_const_vars);
            result.value = cast(type, floor(result.value/denom));
        }
        //std::cout << "  UPPER_BOUND result: " << dim_name << " <= " << result.value << "\n";
    } else {
        assert(false); // Unknown sign, should not have had reached here
    }
    return result;
}

LowerUpperBound convert_nfm_domain_to_lower_upper_bound(
        Type type, NfmDomain& dom, const string& dim_name, int dim_idx,
        const vector<Expr>& sym_const_vars, const vector<Expr>& dim_vars,
        const vector<string> *let_assignments) {
    assert(dim_name.length() >= 0);
    assert(dim_idx >= 0);

    //std::cout << "\nConverting domain " << dom << " to halide\n";

    LowerUpperBound result(dim_name);
    NfmContextDomain& ctx_dom = dom.get_context_domain();
    Expr ctx_cond = convert_nfm_context_domain_to_halide_expr(
        type, ctx_dom, sym_const_vars, let_assignments);
    //std::cout << "  Context cond: " << ctx_cond << "\n";
    if (ctx_cond.defined() && is_zero(ctx_cond)) { // The condition is never true
        return LowerUpperBound(dim_name);
    }
    result.condition = ctx_cond;

    if (dom.is_empty() || dom.is_universe()) {
        return result;
    }

    // Get the lower and upper bounds
    for (auto& eq : dom.get_equalities() ) {
        if (!NfmSolver::nfm_constraint_involves(ctx_dom, eq, dim_idx)) {
            BoundExpr bound = convert_nfm_constraint_to_bound_expr_condition(
                type, ctx_dom, eq, dim_idx, sym_const_vars, dim_vars, let_assignments);
            if (!bound.value.defined()) {
                continue;
            }
            assert(bound.type == CONDITION);
            if (result.condition.defined()) {
                result.condition = And::make(result.condition, bound.value);
            } else {
                result.condition = bound.value;
            }
        } else {
            BoundExpr bound = convert_nfm_constraint_to_bound_expr_non_condition(
                type, ctx_dom, eq, dim_idx, sym_const_vars, dim_vars);
            if (!bound.value.defined()) {
                continue;
            }
            assert(bound.type == EQUAL);
            user_assert(!result.has_lower_bound() && !result.has_upper_bound())
                << "can only have one equality\n";
            // Can have different eq values given different conditions, but
            // within a condition, there is only 1 possible lb/ub value
            result.lb = bound.value;
            result.ub = bound.value;
            result.type = BOUND_EQUALITY;
        }
    }

    for (auto& ineq : dom.get_inequalities()) {
        if (!NfmSolver::nfm_constraint_involves(ctx_dom, ineq, dim_idx)) {
            BoundExpr bound = convert_nfm_constraint_to_bound_expr_condition(
                type, ctx_dom, ineq, dim_idx, sym_const_vars, dim_vars, let_assignments);
            if (!bound.value.defined()) {
                continue;
            }
            assert(bound.type == CONDITION);
            if (result.condition.defined()) {
                result.condition = And::make(result.condition, bound.value);
            } else {
                result.condition = bound.value;
            }
            if (is_zero(result.condition)) { // The condition is never true
                break;
            }
        } else {
            BoundExpr bound = convert_nfm_constraint_to_bound_expr_non_condition(
                type, ctx_dom, ineq, dim_idx, sym_const_vars, dim_vars);
            if (!bound.value.defined()) {
                continue;
            }
            switch (bound.type) {
                case LOWER_BOUND:
                    if (result.type == BOUND_EQUALITY) {
                        // Since we found an equality, it can only have one answer.
                        continue;
                    }
                    if (result.lb.defined()) {
                        result.lb = Max::make(result.lb, bound.value);
                    } else {
                        result.lb = bound.value;
                    }
                    result.type = BOUND_INEQUALITY;
                    break;
                case UPPER_BOUND:
                    if (result.type == BOUND_EQUALITY) {
                        // Since we found an equality, it can only have one answer.
                        continue;
                    }
                    if (result.ub.defined()) {
                        result.ub = Min::make(result.ub, bound.value);
                    } else {
                        result.ub = bound.value;
                    }
                    result.type = BOUND_INEQUALITY;
                    break;
                default:
                    assert(false); // Shouldn't have had reached here
                    break;
            }
        }
    }
    //result.simplify(); // Simplify messed up the lhs/rhs of the let statement
    return result;
}

void convert_to_value_helper(
        const vector<string>& sym_const_vars, const vector<string>& dim_vars,
        vector<Expr>& ands, const Expr& value, map<Expr, Expr, IRDeepCompare>& val_cond,
        const vector<string> *let_assignments) {
    assert(ands.size() > 0);
    size_t n_ineqs_lets = ands.size();
    if (let_assignments != NULL) {
        n_ineqs_lets = std::count_if(ands.begin(), ands.end(),
            [let_assignments] (const Expr& expr) {
                const EQ *eq = expr.as<EQ>();
                if (eq != NULL) {
                    const Variable *var = eq->a.as<Variable>();
                    if (var != NULL) {
                        return !is_val_in_vector(let_assignments, var->name);
                    }
                }
                return true;
        });
        // Sort the let statements based on the dim and sym_const order
        std::sort(ands.begin(), ands.end(),
            [&sym_const_vars, &dim_vars, let_assignments] (const Expr& expr1, const Expr& expr2) {
                const EQ *eq1 = expr1.as<EQ>();
                const EQ *eq2 = expr2.as<EQ>();

                if (!eq1) { // expr1 is an ineq
                    return true; // Ineqs should be sorted before eqs (expr1 < expr2)
                }
                if (!eq2) { // expr2 is an ineq
                    return false; // Ineqs should be sorted before eqs (expr2 < expr1)
                }

                assert(eq1 != NULL);
                assert(eq2 != NULL);
                const Variable *var_eq1 = eq1->a.as<Variable>();
                const Variable *var_eq2 = eq2->a.as<Variable>();
                if ((var_eq1 == NULL) || (var_eq2 == NULL)) {
                    return true;
                }
                if (!is_val_in_vector(let_assignments, var_eq2->name)) {
                    // Eqs (not let expr) should be sorted before the let expr
                    return is_val_in_vector(let_assignments, var_eq1->name);
                }
                return compare_dims(sym_const_vars, dim_vars,
                                    var_eq1->name, var_eq2->name);
        });
    }
    assert(n_ineqs_lets >= 0 && n_ineqs_lets <= ands.size());
    //std::cout << "  n_ineqs_lets: " << n_ineqs_lets << "\n";

    Expr temp1; // cond
    for (size_t i = 0; i < n_ineqs_lets; ++i) {
        if (temp1.defined()) {
            temp1 = And::make(temp1, ands[i]);
        } else {
            temp1 = ands[i];
        }
    }
    //std::cout << "    temp1: " << temp1 << "\n";
    temp1 = simplify(temp1);
    //std::cout << "    simplify temp1: " << temp1 << "\n";

    Expr temp2; // value
    if (n_ineqs_lets == ands.size()) { // No let
        temp2 = value;
    } else {
        temp2 = convert_to_let_helper(ands, n_ineqs_lets, value);
    }
    //std::cout << "    temp2: " << temp2 << "\n";
    temp2 = simplify(temp2);
    //std::cout << "    simplify temp2: " << temp2 << "\n";

    const auto& iter = val_cond.find(temp2);
    if (iter != val_cond.end()) {
        iter->second = Or::make(temp1, iter->second);
    } else {
        val_cond.emplace(temp2, temp1);
    }
}

Expr convert_to_value(const vector<string>& sym_const_vars, const vector<string>& dim_vars,
                    vector<vector<Expr>>& dnf, const vector<Expr>& values, bool is_lower_bound,
                    const vector<string> *let_assignments) {
    assert(dnf.size() > 0);
    assert(dnf.size() == values.size());
    map<Expr, Expr, IRDeepCompare> val_cond; // Map from value to condition
    for (size_t i = 0; i < dnf.size(); ++i) {
        convert_to_value_helper(sym_const_vars, dim_vars, dnf[i],
                                values[i], val_cond, let_assignments);
    }
    assert(val_cond.size() > 0);

    Expr expr;
    if (val_cond.size() == 1) {
        if(!val_cond.begin()->first.defined()) { // All conditions only (ineqs)
            assert(val_cond.begin()->second.defined());
            Expr value = simplify(val_cond.begin()->second);
            //std::cout << "  value1: " << value << "\n";
            expr = value;
        } else {
            assert(val_cond.begin()->first.defined());
            Expr value = simplify(val_cond.begin()->first);
            //std::cout << "  value2: " << value << "\n";
            expr = value;
        }
    } else {
        //std::cout << "\nCONVERT TO LET\n";
        Expr condition, value;
        auto iter = val_cond.rbegin();
        // First element
        assert(iter->first.defined());
        assert(iter->second.defined());
        /*condition = simplify(iter->second);
        value = simplify(iter->first);*/
        condition = iter->second;
        value = iter->first;
        //std::cout << "   cond: " << condition << "; value: " << value << "\n";
        assert(!expr.defined());
        /*if (is_lower_bound) {
            expr = value.type().min();
        } else {
            expr = value.type().max();
        }
        expr = select(condition, value, expr);*/
        expr = value;
        ++iter;

        // Second element
        assert(iter->first.defined());
        assert(iter->second.defined());
        /*condition = simplify(iter->second);
        value = simplify(iter->first);*/
        condition = iter->second;
        value = iter->first;
        //std::cout << "   cond: " << condition << "; value: " << value << "\n";
        assert(expr.defined());
        expr = select(condition, value, expr);
        ++iter;

        for (; iter != val_cond.rend(); ++iter) {
            assert(iter->first.defined());
            assert(iter->second.defined());
            /*condition = simplify(iter->second);
            value = simplify(iter->first);*/
            condition = iter->second;
            value = iter->first;
            //std::cout << "   cond: " << condition << "; value: " << value << "\n";
            assert(expr.defined());
            expr = select(condition, value, expr);
        }
    }
    return expr;
}

Expr convert_to_value(const vector<string>& sym_const_vars, const vector<string>& dim_vars,
                      const Expr& cond, const Expr& value, bool is_lower_bound,
                      const vector<string> *let_assignments) {
    vector<vector<Expr>> dnf = split_expr_into_dnf(cond);
    assert(dnf.size() > 0);
    vector<Expr> values(dnf.size(), value);
    return convert_to_value(sym_const_vars, dim_vars, dnf, values, is_lower_bound, let_assignments);
}

Expr convert_to_value(const vector<string>& sym_const_vars, const vector<string>& dim_vars,
                      const map<Expr, Expr, IRDeepCompare>& bounds, bool is_lower_bound,
                      const vector<string> *let_assignments) {
    vector<vector<Expr>> dnf;
    vector<Expr> values;
    for (auto& iter : bounds) {
        for (auto& ands : split_expr_into_dnf(iter.first)) {
            assert(ands.size() > 0);
            dnf.push_back(std::move(ands));
            values.push_back(iter.second);
        }
    }
    return convert_to_value(sym_const_vars, dim_vars, dnf, values, is_lower_bound, let_assignments);
}

vector<LowerUpperBound> convert_nfm_domain_to_lower_upper_bound_box_helper(
        Type type, NfmDomain& dom, const std::vector<std::string>& box_dims,
        int start_dim_idx,int end_dim_idx, const vector<Expr>& sym_const_vars,
        const vector<Expr>& dim_vars, const vector<string> *let_assignments) {
    assert(box_dims.size() >= 0);
    assert(start_dim_idx <= end_dim_idx);
    assert(start_dim_idx >= 0 && end_dim_idx >= 0);

    //std::cout << "\nConverting domain " << dom << " to halide\n";

    vector<LowerUpperBound> results;
    for (size_t i = 0; i < box_dims.size(); ++i) {
        results.push_back(LowerUpperBound(box_dims[i]));
    }
    NfmContextDomain& ctx_dom = dom.get_context_domain();
    Expr ctx_cond = convert_nfm_context_domain_to_halide_expr(
        type, ctx_dom, sym_const_vars, let_assignments);
    //std::cout << "  Context cond: " << ctx_cond << "\n";
    if (ctx_cond.defined() && is_zero(ctx_cond)) { // The condition is never true
        return results;
    }
    Expr condition = ctx_cond;

    if (dom.is_empty() || dom.is_universe()) {
        for (size_t i = 0; i < results.size(); ++i) {
            results[i].condition = condition;
        }
        return results;
    }

    vector<bool> eq_is_visited(dom.get_num_equalities(), false);
    vector<bool> ineq_is_visited(dom.get_num_inequalities(), false);
    for (int dim_idx = end_dim_idx; dim_idx >= start_dim_idx; --dim_idx) {
        LowerUpperBound& result = results[dim_idx];
        // Get the lower and upper bounds
        for (size_t j = 0; j < dom.get_num_equalities(); ++j) {
            auto& eq = dom.get_equality(j);
            if (eq_is_visited[j] || !NfmSolver::nfm_constraint_involves(ctx_dom, eq, dim_idx)) {
                continue;
            }
            eq_is_visited[j] = true;
            BoundExpr bound = convert_nfm_constraint_to_bound_expr_non_condition(
                type, ctx_dom, eq, dim_idx, sym_const_vars, dim_vars);
            if (!bound.value.defined()) {
                continue;
            }
            assert(bound.type == EQUAL);
            user_assert(!result.has_lower_bound() && !result.has_upper_bound())
                << "can only have one equality\n";
            // Can have different eq values given different conditions, but
            // within a condition, there is only 1 possible lb/ub value
            result.lb = bound.value;
            result.ub = bound.value;
            result.type = BOUND_EQUALITY;
        }

        for (size_t j = 0; j < dom.get_num_inequalities(); ++j) {
            auto& ineq = dom.get_inequality(j);
            if (ineq_is_visited[j] || !NfmSolver::nfm_constraint_involves(ctx_dom, ineq, dim_idx)) {
                continue;
            }
            ineq_is_visited[j] = true;
            BoundExpr bound = convert_nfm_constraint_to_bound_expr_non_condition(
                type, ctx_dom, ineq, dim_idx, sym_const_vars, dim_vars);
            if (!bound.value.defined()) {
                continue;
            }
            switch (bound.type) {
                case LOWER_BOUND:
                    if (result.type == BOUND_EQUALITY) {
                        // Since we found an equality, it can only have one answer.
                        continue;
                    }
                    if (result.lb.defined()) {
                        result.lb = Max::make(result.lb, bound.value);
                    } else {
                        result.lb = bound.value;
                    }
                    result.type = BOUND_INEQUALITY;
                    break;
                case UPPER_BOUND:
                    if (result.type == BOUND_EQUALITY) {
                        // Since we found an equality, it can only have one answer.
                        continue;
                    }
                    if (result.ub.defined()) {
                        result.ub = Min::make(result.ub, bound.value);
                    } else {
                        result.ub = bound.value;
                    }
                    result.type = BOUND_INEQUALITY;
                    break;
                default:
                    assert(false); // Shouldn't have had reached here
                    break;
            }
        }
        //result.simplify(); // Simplify messed up the lhs/rhs of the let statement
    }
    for (size_t i = 0; i < eq_is_visited.size(); ++i) {
        if (!eq_is_visited[i]) {
            auto& eq = dom.get_equality(i);
            BoundExpr bound = convert_nfm_constraint_to_bound_expr_condition(
                type, ctx_dom, eq, start_dim_idx, end_dim_idx, sym_const_vars,
                dim_vars, let_assignments);
            if (!bound.value.defined()) {
                continue;
            }
            assert(bound.type == CONDITION);
            if (condition.defined()) {
                condition = And::make(condition, bound.value);
            } else {
                condition = bound.value;
            }
        }
    }
    for (size_t i = 0; i < ineq_is_visited.size(); ++i) {
        if (!ineq_is_visited[i]) {
            auto& ineq = dom.get_inequality(i);
            BoundExpr bound = convert_nfm_constraint_to_bound_expr_condition(
                type, ctx_dom, ineq, start_dim_idx, end_dim_idx, sym_const_vars,
                dim_vars, let_assignments);
            if (!bound.value.defined()) {
                continue;
            }
            assert(bound.type == CONDITION);
            if (condition.defined()) {
                condition = And::make(condition, bound.value);
            } else {
                condition = bound.value;
            }
        }
    }
    for (size_t i = 0; i < results.size(); ++i) {
        results[i].condition = condition;
    }
    return results;
}

}

Expr convert_nfm_constraint_to_halide_expr(
        Type type, const NfmContextDomain& ctx_dom, const NfmConstraint& cst,
        const vector<string> *let_assignments) {
    const auto& coeff_space = cst.get_coeff_space();
    const auto& space = cst.get_space();
    vector<Expr> sym_const_vars;
    for (size_t i = 0; i < coeff_space.size(); ++i) {
        Expr var = Variable::make(type, coeff_space.get_name(i));
        sym_const_vars.push_back(std::move(var));
    }

    vector<Expr> dim_vars;
    for (size_t i = 0; i < space.size(); ++i) {
        Expr var = Variable::make(type, space.get_name(i));
        dim_vars.push_back(std::move(var));
    }
    return convert_nfm_constraint_to_halide_expr(type, ctx_dom, cst, sym_const_vars,
        dim_vars, let_assignments);
}

Expr convert_nfm_domain_to_halide_expr(Type type, NfmDomain& dom,
                                       const vector<string> *let_assignments) {
    const auto& coeff_space = dom.get_coeff_space();
    const auto& space = dom.get_space();
    vector<Expr> sym_const_vars;
    for (size_t i = 0; i < coeff_space.size(); ++i) {
        Expr var = Variable::make(type, coeff_space.get_name(i));
        sym_const_vars.push_back(std::move(var));
    }

    vector<Expr> dim_vars;
    for (size_t i = 0; i < space.size(); ++i) {
        Expr var = Variable::make(type, space.get_name(i));
        dim_vars.push_back(std::move(var));
    }
    return convert_nfm_domain_to_halide_expr(type, dom, sym_const_vars, dim_vars, let_assignments);
}

Expr convert_nfm_union_domain_to_halide_expr(Type type, NfmUnionDomain& union_dom,
                                             const vector<string> *let_assignments) {
    if (union_dom.is_empty()) {
        return 1 < 0;
    }
    if (union_dom.is_universe()) {
        return 1 > 0;
    }

    const auto& coeff_space = union_dom.get_coeff_space();
    const auto& space = union_dom.get_space();
    vector<Expr> sym_const_vars;
    for (size_t i = 0; i < coeff_space.size(); ++i) {
        Expr var = Variable::make(type, coeff_space.get_name(i));
        sym_const_vars.push_back(std::move(var));
    }

    vector<Expr> dim_vars;
    for (size_t i = 0; i < space.size(); ++i) {
        Expr var = Variable::make(type, space.get_name(i));
        dim_vars.push_back(std::move(var));
    }

    auto& domains = union_dom.get_domains();
    Expr or_expr;
    for (size_t i = 0; i < domains.size(); ++i) {
        Expr expr = convert_nfm_domain_to_halide_expr(type, domains[i],
            sym_const_vars, dim_vars, let_assignments);
        assert(expr.defined());
        if (or_expr.defined()) {
            or_expr = Or::make(or_expr, expr);
        } else {
            or_expr = expr;
        }
    }
    return or_expr;
}

// Convert the union_domain into interval of dim_name, i.e. lb(...) <= dim_name <= ub(...)
Interval convert_nfm_union_domain_to_halide_interval(
        Type type, const Nfm::Internal::NfmUnionDomain& p_union_dom,
        const string& dim_name, const vector<string> *let_assignments) {
    // TODO: Not really clear what should be returned if the union dom is empty.
    // Technically, we could return something like -4 >= x >= 0, but it seems
    // that it's a valid interval in Halide (???)
    if (p_union_dom.is_empty() || p_union_dom.is_universe()) {
        return Interval(dim_name);
    }
    const NfmSpace& coeff_space = p_union_dom.get_coeff_space();
    const NfmSpace& space = p_union_dom.get_space();
    int dim_idx = space.get_index(dim_name);
    if (dim_idx == -1) { // The dimension doesn't exist
        return Interval(dim_name);
    }

    vector<Expr> sym_const_vars;
    for (size_t i = 0; i < coeff_space.size(); ++i) {
        Expr var = Variable::make(type, coeff_space.get_name(i));
        sym_const_vars.push_back(std::move(var));
    }

    vector<Expr> dim_vars;
    for (size_t i = 0; i < space.size(); ++i) {
        Expr var = Variable::make(type, space.get_name(i));
        dim_vars.push_back(std::move(var));
    }

    // Project out everything after this dim so that it doesn't depend on
    // inner loop vars
    NfmUnionDomain union_dom =
        NfmSolver::nfm_union_domain_eliminate_dims(p_union_dom, (size_t)dim_idx+1);
    //std::cout << "Union domain after projection: \n" << union_dom << "\n";

    union_dom = NfmSolver::nfm_union_domain_classify_unknown_coefficient(union_dom, (size_t)dim_idx);

    Interval result(dim_name);
    // Map from lower bound (max of) to condition
    map<Expr, Expr, IRDeepCompare> lower_bounds_temp;
    // Map from upper bound (min of) to condition
    map<Expr, Expr, IRDeepCompare> upper_bounds_temp;
    auto& domains = union_dom.get_domains();
    for (size_t i = 0; i < domains.size(); ++i) { // OR of lower/upper bound (IF-ELSE IF-....-ELSE)
        LowerUpperBound bound = convert_nfm_domain_to_lower_upper_bound(
            type, domains[i], dim_name, dim_idx, sym_const_vars, dim_vars, let_assignments);
        std::cout << "LowerUpperBound bound: condition: " << bound.condition
            << "; lb: " << bound.lb << "; ub: " << bound.ub << "\n";
        if (!bound.is_defined()) {
            continue;
        }
        if (bound.is_always_false()) {
            // The condition is never true
            continue;
        }
        // Lower bound
        if (bound.has_lower_bound()) {
            const auto& iter = lower_bounds_temp.find(bound.lb);
            if (iter != lower_bounds_temp.end()) {
                std::cout << "  lb: " << bound.lb << "; cond: " <<  lower_bounds_temp[bound.lb] << "\n";
                //TODO: might want to revisit this later, to make sure we always get the correct result
                // If condition is undefined, it means universe. Everything
                // OR with universe is a universe
                if (!iter->second.defined() || !bound.condition.defined()) {
                    iter->second = Expr();
                } else {
                    assert(iter->second.defined() && bound.condition.defined());
                    iter->second = Or::make(bound.condition, iter->second);
                }
            } else {
                lower_bounds_temp.emplace(bound.lb, bound.condition);
            }
        }
        // Upper bound
        if (bound.has_upper_bound()) {
            auto iter = upper_bounds_temp.find(bound.ub);
            if (iter != upper_bounds_temp.end()) {
                std::cout << "  ub: " << bound.ub << "; cond: " <<  upper_bounds_temp[bound.ub] << "\n";
                //TODO: might want to revisit this later, to make sure we always get the correct result
                // If condition is undefined, it means universe. Everything
                // OR with universe is a universe
                if (!iter->second.defined() || !bound.condition.defined()) {
                    iter->second = Expr();
                } else {
                    assert(iter->second.defined() && bound.condition.defined());
                    iter->second = Or::make(bound.condition, iter->second);
                }
            } else {
                upper_bounds_temp.emplace(bound.ub, bound.condition);
            }
        }
    }

    //std::cout << "Compressing values with same condition\n";
    // Map from condition to lower bound
    map<Expr, Expr, IRDeepCompare> lower_bounds;
    // Map from condition to upper bound
    map<Expr, Expr, IRDeepCompare> upper_bounds;
    for (auto& it : lower_bounds_temp) { // it (value, condition)
        //std::cout << "  lb: " << it.first << "; cond: " <<  it.second << "\n";
        auto iter = lower_bounds.find(it.second);
        if (iter != lower_bounds.end()) {
            // Halide can only represent box bound; need to take the bounding box
            // (that's why Min of ...)
            iter->second = Min::make(it.first, iter->second);
        } else {
            lower_bounds.emplace(it.second, it.first);
        }
    }
    for (auto& it : upper_bounds_temp) { // it (value, condition)
        //std::cout << "  ub: " << it.first << "; cond: " <<  it.second << "\n";
        auto iter = upper_bounds.find(it.second);
        if (iter != upper_bounds.end()) {
            // Halide can only represent box bound; need to take the bounding box
            // (that's why Max of ...)
            iter->second = Max::make(it.first, iter->second);
        } else {
            upper_bounds.emplace(it.second, it.first);
        }
    }

    // TODO: Assume that they'are disjoint, i.e you can't have universe (undefined)
    // condition and (M > 2 for example) appear together. Assume that OR of
    // all conditions are the universe (which is always true for Halide interval)
    //std::cout << "\nSTART INTERVAL COMPUTATION\n";
    //std::cout << "Computing interval lower bound\n";
    if (lower_bounds.size() == 1) {
        if(!lower_bounds.begin()->first.defined()) {
            // Condition is universe (always true)
            Expr lb = simplify(lower_bounds.begin()->second);
            //std::cout << "  lb: " << lb << "\n";
            result.min = lb;
        } else {
            result.min = convert_to_value(coeff_space.get_names(), space.get_names(),
                                          lower_bounds.begin()->first, lower_bounds.begin()->second,
                                          true, let_assignments);
        }
    } else if (lower_bounds.size() > 1) {
        result.min = convert_to_value(coeff_space.get_names(), space.get_names(),
                                      lower_bounds, true, let_assignments);
    }
    //std::cout << "\nComputing interval upper bound\n";
    if (upper_bounds.size() == 1) {
        if(!upper_bounds.begin()->first.defined()) {
            // Condition is universe (always true)
            Expr ub = simplify(upper_bounds.begin()->second);
            //std::cout << "  ub: " << ub << "\n";
            result.max = ub;
        } else {
            result.max = convert_to_value(coeff_space.get_names(), space.get_names(),
                                          upper_bounds.begin()->first, upper_bounds.begin()->second,
                                          false, let_assignments);
        }
    } else if (upper_bounds.size() > 1) {
        result.max = convert_to_value(coeff_space.get_names(), space.get_names(),
                                      upper_bounds, false, let_assignments);
    }
    //debug(0) << "\nresult.min: " << result.min << "\n";
    //debug(0) << "result.max: " << result.max << "\n";
    result.min = simplify(result.min); // NOTE: Simplify sometimes give odd-looking results
    result.max = simplify(result.max);
    return result;
}

// Convert the union_domain into box.
Box convert_nfm_union_domain_to_halide_box(
        Type type, const Nfm::Internal::NfmUnionDomain& p_union_dom,
        const std::vector<std::string>& box_dims,
        const vector<string> *let_assignments) {
    assert(box_dims.size() > 0);
    if (p_union_dom.is_empty()) {
        return Box();
    }
    if (p_union_dom.is_universe()) {
        Box result;
        result.push_back(Interval(box_dims[0]));
        return result;
    }
    const NfmSpace& coeff_space = p_union_dom.get_coeff_space();
    const NfmSpace& space = p_union_dom.get_space();
    int end_dim_idx = space.get_index(box_dims[box_dims.size()-1]);
    if (end_dim_idx == -1) { // The dimension doesn't exist
        return Box();
    }
    int start_dim_idx = space.get_index(box_dims[0]);
    assert(start_dim_idx != -1);
    assert(end_dim_idx - start_dim_idx + 1 == (int)box_dims.size());

    vector<Expr> sym_const_vars;
    for (size_t i = 0; i < coeff_space.size(); ++i) {
        Expr var = Variable::make(type, coeff_space.get_name(i));
        sym_const_vars.push_back(std::move(var));
    }

    vector<Expr> dim_vars;
    for (size_t i = 0; i < space.size(); ++i) {
        Expr var = Variable::make(type, space.get_name(i));
        dim_vars.push_back(std::move(var));
    }

    // Project out everything after this dim so that it doesn't depend on
    // inner loop vars
    NfmUnionDomain union_dom =
        NfmSolver::nfm_union_domain_eliminate_dims(p_union_dom, (size_t)end_dim_idx+1);
    //std::cout << "Union domain after projection: \n" << union_dom << "\n";

    union_dom = NfmSolver::nfm_union_domain_classify_unknown_coefficient(union_dom, (size_t)end_dim_idx);

    Box result(box_dims.size());
    // Map from lower bound (max of) to condition
    vector<map<Expr, Expr, IRDeepCompare>> lower_bounds_temps(box_dims.size());
    // Map from upper bound (min of) to condition
    vector<map<Expr, Expr, IRDeepCompare>> upper_bounds_temps(box_dims.size());
    auto& domains = union_dom.get_domains();
    for (size_t i = 0; i < domains.size(); ++i) { // OR of lower/upper bound (IF-ELSE IF-....-ELSE)
        vector<LowerUpperBound> bounds = convert_nfm_domain_to_lower_upper_bound_box_helper(
            type, domains[i], box_dims, start_dim_idx, end_dim_idx, sym_const_vars, dim_vars, let_assignments);
        assert(bounds.size() == box_dims.size());

        for (int j = box_dims.size()-1; j >= 0; --j) {
            map<Expr, Expr, IRDeepCompare>& lower_bounds_temp = lower_bounds_temps[j];
            map<Expr, Expr, IRDeepCompare>& upper_bounds_temp = upper_bounds_temps[j];
            LowerUpperBound& bound = bounds[j];
            /*std::cout << "\nDimension (" << j << ")\n";
            std::cout << " LowerUpperBound bound: condition: " << bound.condition
                << "; lb: " << bound.lb << "; ub: " << bound.ub << "\n";*/
            if (!bound.is_defined()) {
                continue;
            }
            if (bound.is_always_false()) {
                // The condition is never true
                continue;
            }
            // Lower bound
            if (bound.has_lower_bound()) {
                const auto& iter = lower_bounds_temp.find(bound.lb);
                if (iter != lower_bounds_temp.end()) {
                    //std::cout << "  lb: " << bound.lb << "; cond: " <<  lower_bounds_temp[bound.lb] << "\n";
                    //TODO: might want to revisit this later, to make sure we always get the correct result
                    // If condition is undefined, it means universe. Everything
                    // OR with universe is a universe
                    if (!iter->second.defined() || !bound.condition.defined()) {
                        iter->second = Expr();
                    } else {
                        assert(iter->second.defined() && bound.condition.defined());
                        iter->second = Or::make(bound.condition, iter->second);
                    }
                } else {
                    lower_bounds_temp.emplace(bound.lb, bound.condition);
                }
            }
            // Upper bound
            if (bound.has_upper_bound()) {
                auto iter = upper_bounds_temp.find(bound.ub);
                if (iter != upper_bounds_temp.end()) {
                    //std::cout << "  ub: " << bound.ub << "; cond: " <<  upper_bounds_temp[bound.ub] << "\n";
                    //TODO: might want to revisit this later, to make sure we always get the correct result
                    // If condition is undefined, it means universe. Everything
                    // OR with universe is a universe
                    if (!iter->second.defined() || !bound.condition.defined()) {
                        iter->second = Expr();
                    } else {
                        assert(iter->second.defined() && bound.condition.defined());
                        iter->second = Or::make(bound.condition, iter->second);
                    }
                } else {
                    upper_bounds_temp.emplace(bound.ub, bound.condition);
                }
            }
        }
    }

    //TODO: how to handle if the domain is not disjoint
    for (int j = box_dims.size()-1; j >= 0; --j) {
        //std::cout << "Compressing values with same condition\n";
        map<Expr, Expr, IRDeepCompare>& lower_bounds_temp = lower_bounds_temps[j];
        map<Expr, Expr, IRDeepCompare>& upper_bounds_temp = upper_bounds_temps[j];
        // Map from condition to lower bound
        map<Expr, Expr, IRDeepCompare> lower_bounds;
        // Map from condition to upper bound
        map<Expr, Expr, IRDeepCompare> upper_bounds;
        //std::cout << "\nDimension (" << j << ")\n";
        for (auto& it : lower_bounds_temp) { // it (value, condition)
            //std::cout << "  lb: " << it.first << "; cond: " << it.second << "\n";
            auto iter = lower_bounds.find(it.second);
            if (iter != lower_bounds.end()) {
                // Halide can only represent box bound; need to take the bounding box
                // (that's why Min of ...)
                iter->second = Min::make(it.first, iter->second);
            } else {
                lower_bounds.emplace(it.second, it.first);
            }
        }
        for (auto& it : upper_bounds_temp) { // it (value, condition)
            //std::cout << "  ub: " << it.first << "; cond: " << it.second << "\n";
            auto iter = upper_bounds.find(it.second);
            if (iter != upper_bounds.end()) {
                // Halide can only represent box bound; need to take the bounding box
                // (that's why Max of ...)
                iter->second = Max::make(it.first, iter->second);
            } else {
                upper_bounds.emplace(it.second, it.first);
            }
        }

        /*for (auto& it : lower_bounds) { // it (cond, value)
            std::cout << "  lb: " << it.second << "; cond: " << it.first << "\n";
        }
        for (auto& it : upper_bounds) { // it (cond, value)
            std::cout << "  ub: " << it.second << "; cond: " << it.first << "\n";
        }*/

        // TODO: Assume that they'are disjoint, i.e you can't have universe (undefined)
        // condition and (M > 2 for example) appear together. Assume that OR of
        // all conditions are the universe (which is always true for Halide interval)
        //std::cout << "\nSTART INTERVAL COMPUTATION\n";
        //std::cout << "Computing interval lower bound\n";
        if (lower_bounds.size() == 1) {
            if(!lower_bounds.begin()->first.defined()) { // Undefined condition is universe (always true)
                Expr lb = simplify(lower_bounds.begin()->second);
                //std::cout << "  lb: " << lb << "\n";
                result[j].min = lb;
            } else {
                result[j].min = convert_to_value(coeff_space.get_names(), space.get_names(),
                                                 lower_bounds.begin()->first, lower_bounds.begin()->second,
                                                 true, let_assignments);
            }
        } else if (lower_bounds.size() > 1) {
            result[j].min = convert_to_value(coeff_space.get_names(), space.get_names(),
                                             lower_bounds, true, let_assignments);
        }
        //std::cout << "\nComputing interval upper bound\n";
        if (upper_bounds.size() == 1) {
            // Undefined condition is universe (always true)
            if(!upper_bounds.begin()->first.defined()) {
                Expr ub = simplify(upper_bounds.begin()->second);
                //std::cout << "  ub: " << ub << "\n";
                result[j].max = ub;
            } else {
                result[j].max = convert_to_value(coeff_space.get_names(), space.get_names(),
                                                 upper_bounds.begin()->first, upper_bounds.begin()->second,
                                                 false, let_assignments);
            }
        } else if (upper_bounds.size() > 1) {
            result[j].max = convert_to_value(coeff_space.get_names(), space.get_names(),
                                             upper_bounds, false, let_assignments);
        }
        //debug(0) << "\nresult[" << j << "].min: " << result[j].min << "\n";
        //debug(0) << "result[" << j << "].max: " << result[j].max << "\n";
        result[j].min = simplify(result[j].min); // NOTE: Simplify sometimes give odd-looking results
        result[j].max = simplify(result[j].max);
    }
    return result;
}

//TODO: Add information on loop dimension. Currently, we treat everything as symbolic constants
Interval nfm_simplify_interval(const Interval& interval) {
    Interval result(interval);
    Expr expr;

    // Simplify lower bound
    expr = convert_interval_to_expr_lower_bound(interval);
    //std::cout << "  Simplify lower bound: " << expr << "\n";
    if (expr.defined()) {
        CollectVars collect({interval.var});
        collect.mutate(expr);
        const auto& let_assignments = collect.get_let_assignments();
        NfmUnionDomain union_dom = convert_halide_expr_to_nfm_union_domain(
            expr, collect.get_sym_consts(), collect.get_dims());
        Interval temp =
            convert_nfm_union_domain_to_halide_interval(Int(32), union_dom, interval.var, &let_assignments);
        //std::cout << "    After simplifying lower bound: " << temp.min << "\n";
        result.min = temp.min;
    }

    // Simplify upper bound
    expr = convert_interval_to_expr_upper_bound(interval);
    //std::cout << "  Simplify upper bound: " << expr << "\n";
    if (expr.defined()) {
        CollectVars collect({interval.var});
        collect.mutate(expr);
        const auto& let_assignments = collect.get_let_assignments();
        NfmUnionDomain union_dom = convert_halide_expr_to_nfm_union_domain(
            expr, collect.get_sym_consts(), collect.get_dims());
        Interval temp =
            convert_nfm_union_domain_to_halide_interval(Int(32), union_dom, interval.var, &let_assignments);
        //std::cout << "    After simplifying upper bound: " << temp.max << "\n";
        result.max = temp.max;
    }
    return result;
}

//TODO: Add information on loop dimension. Currently, we treat everything as symbolic constants
//TODO: infer the type from expr
Expr nfm_simplify_expr(const Expr& expr) {
    //std::cout << "\nnfm_simplify_expr: " << expr << "\n";
    if (!expr.defined()) {
        return expr;
    }
    Expr result;
    CollectVars collect;
    collect.mutate(expr);
    const auto& let_assignments = collect.get_let_assignments();
    NfmUnionDomain union_dom = convert_halide_expr_to_nfm_union_domain(
        expr, collect.get_sym_consts(), collect.get_dims());

    /*debug(0) << "  Union Domain (AFTER SIMPLIFY using NFM) (" << union_dom.get_domains().size() << "): \n";
    union_dom.sort();
    for (const auto& dom : union_dom.get_domains()) {
        debug(0) << "  " << dom << "\n";
        debug(0) << "    Context: " << dom.get_context_domain() << "\n";
    }
    std::cout << "is universe? " << union_dom.is_universe() << "\n";
    std::cout << "is empty? " << union_dom.is_empty() << "\n";*/

    result = convert_nfm_union_domain_to_halide_expr(Int(32), union_dom, &let_assignments);
    //std::cout << "Result nfm_simplify_expr: " << result << "\n";
    return result;
}

}
}
