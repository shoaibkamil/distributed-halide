#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include "nfm_solver.h"

#include "nfm_domain.h"

namespace Nfm {
namespace Internal {

using std::ostringstream;
using std::string;
using std::vector;

// Return the constraint opposite to cst, e.g if cst is u >= 0, return u < 0
// (u <= -1 or u+1 >= 0)
NfmConstraint NfmSolver::nfm_constraint_get_opposite(const NfmConstraint& cst) {
    NfmConstraint opposite = cst.neg() - 1;
    return opposite;
}

// Return true if constraint involves variable var.
bool NfmSolver::nfm_constraint_involves(
        const NfmContextDomain& ctx_dom, NfmConstraint& cst, const string& var) {
    int var_idx = cst.get_space().get_index(var);
    return nfm_constraint_involves(ctx_dom, cst, var_idx);
}

bool NfmSolver::nfm_constraint_involves(
        const NfmContextDomain& ctx_dom, NfmConstraint& cst, int var_idx) {
    if (var_idx < 0) { // The dimension doesn't exist
        return false;
    }
    // If coeff is unknown, we'll still consider it as true
    NfmPolyCoeff& coeff = cst.get_coeff(var_idx);
    return !nfm_poly_coeff_is_zero(ctx_dom, coeff);
}

bool NfmSolver::nfm_constraint_is_lower_bound(
        const NfmContextDomain& ctx_dom, NfmConstraint& cst, const string& var) {
    int var_idx = cst.get_space().get_index(var);
    return nfm_constraint_is_lower_bound(ctx_dom, cst, var_idx);
}

bool NfmSolver::nfm_constraint_is_lower_bound(
        const NfmContextDomain& ctx_dom, NfmConstraint& cst, int var_idx) {
    if (var_idx == -1) { // The dimension doesn't exist if we get -1
        return false;
    }
    if (cst.is_equality()) { // If it's an equality, it's both lower and upper bound
        return true;
    }
    NfmPolyCoeff& coeff = cst.get_coeff(var_idx);
    return nfm_poly_coeff_is_pos(ctx_dom, coeff);
}

bool NfmSolver::nfm_constraint_is_upper_bound(
        const NfmContextDomain& ctx_dom, NfmConstraint& cst, const string& var) {
    int var_idx = cst.get_space().get_index(var);
    return nfm_constraint_is_upper_bound(ctx_dom, cst, var_idx);
}

bool NfmSolver::nfm_constraint_is_upper_bound(
        const NfmContextDomain& ctx_dom, NfmConstraint& cst, int var_idx) {
    if (var_idx < 0) { // The dimension doesn't exist
        return false;
    }
    if (cst.is_equality()) { // If it's an equality, it's both lower and upper bound
        return true;
    }
    NfmPolyCoeff& coeff = cst.get_coeff(var_idx);
    return nfm_poly_coeff_is_neg(ctx_dom, coeff);
}

NfmSign NfmSolver::nfm_poly_coeff_get_sign(const NfmContextDomain& ctx_dom,
                                           NfmPolyCoeff& coeff) {
    if (!coeff.is_unknown()) {
        return coeff.sign_;
    }
    if (coeff.is_linear()) {
        coeff.sign_ = nfm_poly_coeff_linear_get_sign(ctx_dom, coeff);
    } else {
        coeff.sign_ = nfm_poly_coeff_non_linear_get_sign(ctx_dom, coeff);
    }
    return coeff.sign_;
}

bool NfmSolver::nfm_poly_coeff_is_pos(const NfmContextDomain& ctx_dom,
                                      NfmPolyCoeff& coeff) {
    return (nfm_poly_coeff_get_sign(ctx_dom, coeff) == NFM_POSITIVE);
}

bool NfmSolver::nfm_poly_coeff_is_neg(const NfmContextDomain& ctx_dom,
                                      NfmPolyCoeff& coeff) {
    return (nfm_poly_coeff_get_sign(ctx_dom, coeff) == NFM_NEGATIVE);
}

bool NfmSolver::nfm_poly_coeff_is_zero(const NfmContextDomain& ctx_dom,
                                       NfmPolyCoeff& coeff) {
    return (nfm_poly_coeff_get_sign(ctx_dom, coeff) == NFM_ZERO);
}

bool NfmSolver::nfm_poly_coeff_is_unknown(const NfmContextDomain& ctx_dom,
                                          NfmPolyCoeff& coeff) {
    return (nfm_poly_coeff_get_sign(ctx_dom, coeff) == NFM_UNKNOWN);
}

// Coeff has to be linear otherwise no guarantee on the result
NfmSign NfmSolver::nfm_poly_coeff_linear_get_sign(
        const NfmContextDomain& ctx_dom, const NfmPolyCoeff& coeff) {
    assert(ctx_dom.get_space() == coeff.get_space());
    assert(coeff.is_linear());

    // First assume that coeff >= 0
    NfmContextDomain context_dom1(ctx_dom);
    context_dom1.add_context(NfmContext(coeff, false)); // Add coeff >= 0
    context_dom1.simplify();
    if (context_dom1.is_empty()) {
        // Since coeff >= 0 isn't possible, it must have been negative
        return NFM_NEGATIVE;
    }

    // Assume that coeff <= 0
    NfmContextDomain context_dom2(ctx_dom);
    context_dom2.add_context(NfmContext(-coeff, false)); // Add coeff <= 0
    context_dom2.simplify();
    if (context_dom2.is_empty()) {
        // Since coeff <= 0 isn't possible, it must have been positive
        return NFM_POSITIVE;
    }
    return NFM_UNKNOWN;
}

// Coeff has to be non-linear otherwise no guarantee on the result
 NfmSign NfmSolver::nfm_poly_coeff_non_linear_get_sign(
        const NfmContextDomain& ctx_dom, const NfmPolyCoeff& coeff) {
    assert(ctx_dom.get_space() == coeff.get_space());
    assert(!coeff.is_linear());

    // TODO: The search is not comprehensive, e.g. we ignore the case when context
    // a-b >= 5 and c >= 0 and symbolic coeff is a-b+c. We could have determined
    // a-b+c as positive in this case, but we don't do it (have to search too many comb)
    bool found_pair = false;
    for (const auto& ctx : ctx_dom.get_non_linear_contexts()) {
        int context_const = ctx.get_constant();
        int coeff_const = coeff.get_constant();

        const NfmPolyCoeff& ctx_no_const = ctx.get_context() - context_const;
        const NfmPolyCoeff& coeff_no_const = coeff - coeff_const;

        if (ctx_no_const == coeff_no_const) {
            if (coeff_const > context_const) {
                return NFM_POSITIVE;
            } else if (coeff_const == context_const) {
                if (found_pair) {
                    return NFM_ZERO;
                } else {
                    found_pair = true;
                }
            }
        }
        if (-ctx_no_const == coeff_no_const) {
            if (-coeff_const > context_const) {
                return NFM_NEGATIVE;
            } else if (coeff_const == -context_const) {
                if (found_pair) {
                    return NFM_ZERO;
                } else {
                    found_pair = true;
                }
            }
        }
    }
    return NFM_UNKNOWN;
}

// Given an instance of NfmConstraint 'p_cst', which coeff's sign at 'pos' is
// unknown, return a list of instances of NfmConstraint with (+, -, 0) sign
// in that order
vector<NfmSolver::NfmContextConstraint> NfmSolver::nfm_constraint_classify_unknown_coefficient(
        const NfmConstraint& p_cst, size_t pos) {
    assert(pos < p_cst.get_space().size());
    const NfmPolyCoeff& coeff = p_cst[pos];

    vector<NfmContextConstraint> result;
    // Positive
    // a > 0 equal to a - 1 >= 0
    result.push_back({NfmContext(coeff-1, false),
                      p_cst.set_coeff_sign(pos, NFM_POSITIVE)});

    // Negative
    // a < 0 equal to -a - 1 >= 0
    result.push_back({NfmContext(-coeff-1, false),
                      p_cst.set_coeff_sign(pos, NFM_NEGATIVE)});

    // Zero
    result.push_back({NfmContext(coeff, true),
                      p_cst.set_coeff_sign(pos, NFM_ZERO)});
    return result;
}

vector<NfmDomain> NfmSolver::nfm_domain_classify_unknown_coefficient_helper(
        NfmDomain& p_dom, size_t pos) {
    assert(pos < p_dom.space_.size());
    if (p_dom.is_empty() || p_dom.is_universe()) {
        return {p_dom};
    }

    const NfmContextDomain& ctx_dom = p_dom.get_context_domain();

    NfmDomain temp_dom(p_dom.coeff_space_, p_dom.space_, p_dom.get_context_domain());
    vector<vector<NfmContextConstraint>> classified_constraints;
    for (size_t i = 0; i < p_dom.get_num_equalities(); ++i) {
        if (nfm_poly_coeff_is_unknown(ctx_dom, p_dom.eqs_[i][pos])) {
            vector<NfmContextConstraint> classified_cst =
                nfm_constraint_classify_unknown_coefficient(p_dom.eqs_[i], pos);
            classified_constraints.push_back(std::move(classified_cst));
        } else {
            temp_dom.add_constraint(p_dom.eqs_[i]);
        }
    }
    for (size_t i = 0; i < p_dom.get_num_inequalities(); ++i) {
        if (nfm_poly_coeff_is_unknown(ctx_dom, p_dom.ineqs_[i][pos])) {
            vector<NfmContextConstraint> classified_cst =
                nfm_constraint_classify_unknown_coefficient(p_dom.ineqs_[i], pos);
            classified_constraints.push_back(std::move(classified_cst));
        } else {
            temp_dom.add_constraint(p_dom.ineqs_[i]);
        }
    }

    vector<NfmDomain> result = {temp_dom};
    for (const auto& csts : classified_constraints) {
        vector<NfmDomain> temp;
        for (const NfmDomain& dom : result) {
            for (const NfmContextConstraint& ctx_cst : csts) {
                NfmDomain new_dom(dom);
                new_dom.add_constraint(ctx_cst.constraint);
                new_dom.add_context(ctx_cst.context);
                temp.push_back(std::move(new_dom));
            }
        }
        result = temp;
    }
    return result;
}

NfmUnionDomain NfmSolver::nfm_domain_classify_unknown_coefficient(
        NfmDomain& p_dom, size_t pos) {
    NfmUnionDomain union_dom(p_dom.coeff_space_, p_dom.space_);
    if (p_dom.is_empty() || p_dom.is_universe()) {
        return union_dom;
    }
    if (pos >= union_dom.space_.size()) {
        union_dom.add_domain(p_dom);
        return union_dom;
    }

    vector<NfmDomain> classified_doms =
        nfm_domain_classify_unknown_coefficient_helper(p_dom, pos);

    for (const NfmDomain& dom : classified_doms) {
        if(dom.is_empty()) {
            continue;
        }
        union_dom.add_domain(std::move(dom));
    }
    return union_dom;
}

NfmUnionDomain NfmSolver::nfm_union_domain_classify_unknown_coefficient(
        NfmUnionDomain& p_union_dom, size_t pos) {
    if (pos >= p_union_dom.space_.size()) {
        return p_union_dom;
    }
    if (p_union_dom.is_empty() || p_union_dom.is_universe()) {
        return p_union_dom;
    }

    NfmUnionDomain union_dom(p_union_dom.coeff_space_, p_union_dom.space_);
    vector<NfmDomain> temp;
    for (NfmDomain& dom : p_union_dom.domains_) {
        vector<NfmDomain> classified_doms =
            nfm_domain_classify_unknown_coefficient_helper(dom, pos);
        for (NfmDomain& cdom : classified_doms) {
            if (cdom.is_empty()) {
                continue;
            }
            temp.push_back(std::move(cdom));
        }
    }

    for (const NfmDomain& dom : temp) {
        assert(!dom.is_empty());
        union_dom.add_domain(std::move(dom));
    }
    return union_dom;
}

void NfmSolver::nfm_domain_swap_equality(NfmDomain& dom, int a, int b) {
    assert(a < (int)dom.eqs_.size());
    assert(b < (int)dom.eqs_.size());
    std::iter_swap(dom.eqs_.begin()+a, dom.eqs_.begin()+b);
}

void NfmSolver::nfm_domain_swap_inequality(NfmDomain& dom, int a, int b) {
    assert(a < (int)dom.ineqs_.size());
    assert(b < (int)dom.ineqs_.size());
    std::iter_swap(dom.ineqs_.begin()+a, dom.ineqs_.begin()+b);
}

// Remove equalities at position "pos" from the domain.
// Side effect: move the equality at the back of the eqs_ vector to pos "pos"
int NfmSolver::nfm_domain_drop_equality(NfmDomain& dom, size_t pos) {
    if (pos >= dom.get_num_equalities()) {
       return -1;
    }
    //printf("  Dropping equality (pos: %d): %s\n", (int)pos, dom.eqs_[pos].to_string().c_str());
    if (pos != dom.get_num_equalities() - 1) {
       size_t back = dom.get_num_equalities()-1;
       std::iter_swap(dom.eqs_.begin()+pos, dom.eqs_.begin()+back);
    }
    dom.eqs_.pop_back();
    return 0;
}

// Remove inequalities at position "pos" from the domain.
// Side effect: move the inequality at the back of the ineqs vector to pos "pos"
int NfmSolver::nfm_domain_drop_inequality(NfmDomain& dom, size_t pos) {
    if (pos >= dom.get_num_inequalities()) {
        return -1;
    }
    //printf("  Dropping inequality (pos: %d): %s\n", (int)pos, dom.ineqs_[pos].to_string().c_str());
    if (pos != dom.get_num_inequalities() - 1) {
       size_t back = dom.get_num_inequalities()-1;
       std::iter_swap(dom.ineqs_.begin()+pos, dom.ineqs_.begin()+back);
    }
    dom.ineqs_.pop_back();
    return 0;
}

void NfmSolver::nfm_domain_inequality_to_equality(NfmDomain& dom, size_t pos1, size_t pos2) {
    assert(pos1 < dom.get_num_inequalities());
    assert(pos2 < dom.get_num_inequalities());
    assert(pos1 != pos2);

    const NfmConstraint& ineq = dom.ineqs_[pos1];
    NfmConstraint eq = ineq.to_equality();
    dom.add_constraint(std::move(eq));

    size_t orig_size = dom.get_num_inequalities();
    nfm_domain_drop_inequality(dom, pos1);
    if (pos2 != orig_size-1) {
        nfm_domain_drop_inequality(dom, pos2);
    } else {
        // We have moved pos2 (from the end of the list) to pos1 during the first drop
        nfm_domain_drop_inequality(dom, pos1);
    }
}

// Given two instances of NfmConstraint, eliminate the dim at index pos
NfmConstraint NfmSolver::nfm_constraint_elim_helper(
        const NfmContextDomain& ctx_dom, const NfmConstraint& pos_cst,
        const NfmConstraint& neg_cst, size_t pos) {

    const NfmPolyCoeff& coeff_pos = pos_cst[pos];
    const NfmPolyCoeff& coeff_neg = neg_cst[pos];

    // Special case when coeff_pos + coeff_neg = 0
    NfmPolyCoeff coeff_sum = coeff_pos + coeff_neg;
    if (nfm_poly_coeff_is_zero(ctx_dom, coeff_sum)) {
        return pos_cst + neg_cst;
    }

    int gcd = non_neg_gcd(coeff_pos.content(), coeff_neg.content());

    // Multiply the positive Constraint by -px_neg/gcd_val
    NfmPolyCoeff pos_mul = -coeff_neg.exquo(gcd);
    NfmConstraint new_pos_cst = pos_cst.mul(pos_mul);

    // Multiply the negatives Constraint by px_pos/gcd_val
    NfmPolyCoeff neg_mul = coeff_pos.exquo(gcd);
    NfmConstraint new_neg_cst = neg_cst.mul(neg_mul);

    // Sum the new inequalities together to eliminate dim at pos
    return new_pos_cst + new_neg_cst;
}

// Eliminate dst's coeff at pos if applicable using src as reference
NfmConstraint NfmSolver::nfm_constraint_elim(
        const NfmContextDomain& ctx_dom, NfmConstraint& dst,
        NfmConstraint& src, size_t pos) {
    assert(dst.get_space() == src.get_space());
    assert(dst.get_coeff_space() == src.get_coeff_space());
    // At this point, both dst[pos] and src[pos] can't be unknown
    assert(!nfm_poly_coeff_is_unknown(ctx_dom, dst[pos]));
    assert(!nfm_poly_coeff_is_unknown(ctx_dom, src[pos]));

    if (nfm_poly_coeff_is_zero(ctx_dom, dst[pos])) {
        return dst;
    }
    if (nfm_poly_coeff_is_pos(ctx_dom, dst[pos])) {
        if (nfm_poly_coeff_is_neg(ctx_dom, src[pos])) {
            return nfm_constraint_elim_helper(ctx_dom, dst, src, pos);
        }
        return nfm_constraint_elim_helper(ctx_dom, dst, -src, pos);
    } else {
        if (nfm_poly_coeff_is_neg(ctx_dom, src[pos])) {
            return nfm_constraint_elim_helper(ctx_dom, -src, dst, pos);
        }
        return nfm_constraint_elim_helper(ctx_dom, src, dst, pos);
    }
}

// Try to eliminate dim at pos from the remaining ineqs/eqs of the domain using
// 'eq' as pivot. Return True if there are no invalid eqs/ineqs; return False
// otherwise.
bool NfmSolver::nfm_domain_eliminate_var_using_equality(
        NfmDomain& dom, size_t pos, NfmConstraint& eq, int *progress) {
    assert(pos < eq.get_space().size());

    NfmContextDomain& ctx_dom = dom.get_context_domain();
    assert(!nfm_poly_coeff_is_zero(ctx_dom, eq[pos]));
    assert(!nfm_poly_coeff_is_unknown(ctx_dom, eq[pos]));

    if (progress) {
        *progress = 0;
    }
    for (size_t k = 0; k < dom.get_num_equalities(); ++k) {
        if (dom.eqs_[k] == eq) {
            continue;
        }
        NfmPolyCoeff& coeff = dom.eqs_[k][pos];
        if (nfm_poly_coeff_is_zero(ctx_dom, coeff)) {
            continue;
        } else if (nfm_poly_coeff_is_unknown(ctx_dom, coeff)) {
            // Can't do anything if the sign of coeff at pos is unknown
            continue;
        }
        if (progress) {
            *progress = 1;
        }
        NfmConstraint elim_eq = nfm_constraint_elim(ctx_dom, dom.eqs_[k], eq, pos);
        /*printf("  Eliminate (dim: %s):\n    eq  %s \n    eq  %s \n    Res:  %s\n",
               dom.get_space()[pos].c_str(), dom.eqs_[k].to_string().c_str(),
               eq.to_string().c_str(), elim_eq.to_string().c_str());*/
        if (elim_eq.is_infeasible()) {
            return false;
        }
        if (elim_eq.is_constant() && (!nfm_poly_coeff_is_zero(ctx_dom, elim_eq.get_constant()))) {
            // Update context: constant == 0
            dom.add_context(NfmContext(elim_eq.get_constant(), true));
        }
        dom.eqs_[k] = elim_eq;
    }

    for (size_t k = 0; k < dom.get_num_inequalities(); ++k) {
        NfmPolyCoeff& coeff = dom.ineqs_[k][pos];
        if (nfm_poly_coeff_is_zero(ctx_dom, coeff)) {
            continue;
        } else if (nfm_poly_coeff_is_unknown(ctx_dom, coeff)) {
            // Can't do anything if the sign of coeff at pos is unknown
            continue;
        }
        if (progress) {
            *progress = 1;
        }
        NfmConstraint elim_ineq = nfm_constraint_elim(ctx_dom, dom.ineqs_[k], eq, pos);
        /*printf("  Eliminate (dim: %s):\n    ineq  %s \n    eq  %s \n    Res:  %s\n",
               dom.get_space()[pos].c_str(), dom.ineqs_[k].to_string().c_str(),
               eq.to_string().c_str(), elim_ineq.to_string().c_str());*/
        if (elim_ineq.is_infeasible()) {
            return false;
        }
        if (elim_ineq.is_constant() && (!nfm_poly_coeff_is_zero(ctx_dom, elim_ineq.get_constant()))) {
            // Update context: constant >= 0
            dom.add_context(NfmContext(elim_ineq.get_constant(), false));
        }
        dom.ineqs_[k] = elim_ineq;
    }
    return true;
}

// Divide equalities and inequalities by their gcd
NfmDomain NfmSolver::nfm_domain_normalize_constraints(NfmDomain& dom) {
    if (dom.is_empty() || dom.is_universe()) {
        return dom;
    }
    NfmDomain new_dom(dom.get_coeff_space(), dom.get_space(), dom.get_context_domain());
    NfmContextDomain& ctx_dom = new_dom.get_context_domain();
    for (int i = 0; i < (int)dom.get_num_equalities(); ++i) {
        NfmConstraint eq = dom.get_equality(i).simplify();
        NfmPolyCoeff& constant = eq.get_constant();
        int dim_gcd = eq.non_constant_content(); // Integer GCD of non-constant terms
        if (dim_gcd == 0) { // It only has (symbolic) constant term
            if (nfm_poly_coeff_is_unknown(ctx_dom, constant)) {
                // Update context: constant == 0
                new_dom.add_context(NfmContext(constant, true));
                //printf("  Adding new constraint to context: %s\n", eq.to_string().c_str());
                continue;
            } else if (!nfm_poly_coeff_is_zero(ctx_dom, constant)) { // Infeasible domain
                //printf("  Infeasible Equality: coeffs of dims are all zero, but constant is non-zero\n");
                return NfmDomain::empty_domain(dom.get_coeff_space(), dom.get_space());
            }
            continue;
        }
        // We can't really do anything if the constant term is a polynomial
        if (!constant.is_constant()) {
            new_dom.add_constraint(eq);
            continue;
        }
        int gcd = non_neg_gcd(dim_gcd, constant.content()); // GCD of the whole terms' coeffs
        if (gcd == 1) {
            new_dom.add_constraint(eq);
            continue;
        }
        if (gcd != dim_gcd) { // Constant is not divisible by dim_gcd
            //printf("  Infeasible Equality: coeff of constant term is not divisible by dim_gcd\n");
            return NfmDomain::empty_domain(dom.get_coeff_space(), dom.get_space());
        }
        new_dom.add_constraint(eq.exquo(gcd));
    }

    for (int i = 0; i < (int)dom.get_num_inequalities(); ++i) {
        NfmConstraint ineq = dom.get_inequality(i).simplify();
        NfmPolyCoeff& constant = ineq.get_constant();
        int dim_gcd = ineq.non_constant_content(); // Integer GCD of non-constant terms
        if (dim_gcd == 0) { // It only has (symbolic) constant term
            if (nfm_poly_coeff_is_unknown(ctx_dom, constant)) {
                // Update context: constant >= 0
                new_dom.add_context(NfmContext(constant, false));
                //printf("  Adding new constraint to context: %s\n", ineq.to_string().c_str());
                continue;
            } else if (nfm_poly_coeff_is_neg(ctx_dom, constant)) { // Infeasible domain
                //printf("  Infeasible Inequality: coeffs of dims are all zero, but constant is negative\n");
                return NfmDomain::empty_domain(dom.get_coeff_space(), dom.get_space());
            }
            continue;
        }
        // We can't really do anything if the constant term is a polynomial
        if (!constant.is_constant()) {
            new_dom.add_constraint(ineq);
            continue;
        }
        int gcd = non_neg_gcd(dim_gcd, constant.content()); // GCD of the whole terms' coeffs
        new_dom.add_constraint(ineq.fdiv(gcd));
    }
    return new_dom;
}

// Return a tuple of new domain where all the duplicates are removed and a boolean
// indicating whether there are duplicates. If we find implicit equalities in
// inequalities contraints, we convert it into an explicit equality.
NfmDomain NfmSolver::nfm_domain_remove_duplicate_constraints(NfmDomain& dom, int *progress) {
    if (progress) {
        *progress = 0;
    }
    if (dom.is_empty() || dom.is_universe()) {
        return dom;
    }

    // Return index of (in-)equality with same non-constant terms' coeffs as
    // (in-)eq at index k. Return -1 if there isn't any.
    auto get_duplicate_index = [](const vector<NfmConstraint>& constraints, int k) {
        NfmConstraint cst = constraints[k] - constraints[k].get_constant();
        for (int i = 0; i < (int)constraints.size(); ++i) {
            if (k == i) {
                continue;
            }
            NfmConstraint other_cst = constraints[i] - constraints[i].get_constant();
            if (cst == other_cst) {
                return i;
            }
        }
        return -1;
    };

    NfmContextDomain& ctx_dom = dom.get_context_domain();
    for (int k = 0; k < (int)dom.get_num_inequalities(); ++k) {
        NfmConstraint& ineq = dom.ineqs_[k];
        if (ineq.is_constant()) {
            NfmPolyCoeff& constant = ineq.get_constant();
            if (nfm_poly_coeff_is_pos(ctx_dom, constant) ||
                    nfm_poly_coeff_is_zero(ctx_dom, constant)) {
                nfm_domain_drop_inequality(dom, k);
                if (k > 1) {
                    k -= 1;
                }
                continue;
            } else if (nfm_poly_coeff_is_neg(ctx_dom, constant)) {
                /*printf("  Infeasible domain: ineq dims' coeffs are all zero but "
                       "the constant term is negative\n");*/
                return NfmDomain::empty_domain(dom.get_coeff_space(), dom.get_space());
            } else {
                // Update context: constant >= 0
                dom.add_context(NfmContext(constant, false));
                nfm_domain_drop_inequality(dom, k);
                if (k > 1) {
                    k -= 1;
                }
                continue;
            }
        }
        int l = get_duplicate_index(dom.ineqs_, k);
        if (l == -1) {
            continue;
        }
        NfmConstraint& other_ineq = dom.ineqs_[l];
        NfmSign sign = ineq.get_constant().compare(other_ineq.get_constant());
        if (sign == NFM_UNKNOWN) { // other_ineq.constant ? ineq.constant
            continue;
        }
        //printf("    %s subsumes %s\n", ineq.to_string().c_str(), other_ineq.to_string().c_str());
        if (progress) {
            *progress = 1;
        }
        if ((sign == NFM_NEGATIVE) || (sign == NFM_ZERO)) {
            // ineq.constant < other.constant or other_ineq.constant < other_ineq.constant
            // (We're keeping ineq and dropping other_ineq)
            nfm_domain_swap_inequality(dom, k, l);
        }
        nfm_domain_drop_inequality(dom, k);
        if (k > 1) {
            k -= 1;
        }
    }

    for (int k = 0; k < (int)dom.get_num_inequalities()-1; ++k) {
        NfmConstraint temp = dom.ineqs_[k];
        dom.ineqs_[k] = -dom.ineqs_[k];
        int l = get_duplicate_index(dom.ineqs_, k);
        dom.ineqs_[k] = temp;
        if (l == -1) {
            continue;
        }
        NfmPolyCoeff constant_sum = dom.ineqs_[k].get_constant() + dom.ineqs_[l].get_constant();
        if (nfm_poly_coeff_is_zero(ctx_dom, constant_sum)) { // Convert into equality
            /*printf("  Converting %s and %s into equality\n", dom.ineqs_[k].to_string().c_str(),
                   dom.ineqs_[l].to_string().c_str());*/
            if (progress) {
                *progress = 1;
            }
            nfm_domain_inequality_to_equality(dom, k, l);
            if (k > 1) {
                k -= 1;
            }
        } else if (nfm_poly_coeff_is_neg(ctx_dom, constant_sum)) {
            /*printf("  %s and %s, constant_sum: %s\n", dom.ineqs_[k].to_string().c_str(),
                   dom.ineqs_[l].to_string().c_str(), constant_sum.to_string().c_str());*/
            //printf("  Infeasible domain, constant sum is not non-negative/unknown\n");
            return NfmDomain::empty_domain(dom.get_coeff_space(), dom.get_space());
        }
    }

    for (int k = 0; k < (int)dom.get_num_equalities(); ++k) {
        NfmConstraint& eq = dom.eqs_[k];
        if (eq.is_constant()) {
            NfmPolyCoeff& constant = eq.get_constant();
            if (nfm_poly_coeff_is_zero(ctx_dom, constant)) {
                nfm_domain_drop_equality(dom, k);
                if (k > 1) {
                    k -= 1;
                }
                continue;
            } else if (nfm_poly_coeff_is_unknown(ctx_dom, constant)) {
                // Update context: constant == 0
                dom.add_context(NfmContext(constant, true));
                nfm_domain_drop_equality(dom, k);
                if (k > 1) {
                    k -= 1;
                }
                continue;
            } else {
                /*printf("  Infeasible domain: eq dims' coeffs are all zero but "
                       "the constant term is non-zero\n");*/
                return NfmDomain::empty_domain(dom.get_coeff_space(), dom.get_space());
            }
        }
        int l = get_duplicate_index(dom.eqs_, k);
        if (l == -1) {
            continue;
        }
        const NfmConstraint& other_eq = dom.eqs_[l];
        if (eq.get_constant() != other_eq.get_constant()) {
            //printf("  Infeasible domain, found 2 eqs with the same dims' coeffs but different constant\n");
            return NfmDomain::empty_domain(dom.get_coeff_space(), dom.get_space());
        }
        if (progress) {
            *progress = 1;
        }
        nfm_domain_drop_equality(dom, k);
        if (k > 1) {
            k -= 1;
        }
    }

    for (size_t i = 0; i < dom.get_num_inequalities(); ++i) {
        NfmConstraint temp = dom.ineqs_[i];
        dom.ineqs_[i] = nfm_constraint_get_opposite(dom.ineqs_[i]);
        NfmUnionDomain elim_dom = nfm_domain_eliminate_dims(dom, 0);
        dom.ineqs_[i] = temp;
        if (elim_dom.is_empty()) {
            //printf("  Redundant ineq constraint: %s\n", temp.to_string().c_str());
            nfm_domain_drop_inequality(dom, i);
            if (i > 1) {
                i -= 1;
            }
        }
    }
    return dom;
}

bool NfmSolver::nfm_constraint_ineq_is_redundant(const NfmDomain& p_dom,
                                                 const NfmConstraint& cst) {
    assert(!cst.is_equality());
    NfmDomain dom(p_dom);
    dom.ineqs_.push_back(nfm_constraint_get_opposite(cst));
    NfmUnionDomain elim_dom = nfm_domain_eliminate_dims(dom, 0);
    return elim_dom.is_empty();
}

bool NfmSolver::nfm_constraint_ineq_is_redundant(const NfmUnionDomain& p_union_dom,
                                                 const NfmConstraint& cst) {
    assert(!cst.is_equality());
    for (const auto& dom : p_union_dom.domains_) {
        if (nfm_constraint_ineq_is_redundant(dom, cst)) {
            return true;
        }
    }
    return false;
}

NfmDomain NfmSolver::nfm_domain_gauss(NfmDomain& dom, int *progress) {
    if (progress) {
        *progress = 0;
    }
    if (dom.is_empty() || dom.is_universe()) {
        return dom;
    }
    NfmContextDomain& ctx_dom = dom.get_context_domain();
    int last_var = dom.space_.size() - 1;
    int done = 0;
    int k = 0;
    for (done = 0; done < (int)dom.get_num_equalities(); ++done) {
        for (; last_var >= 0; --last_var) {
            // Find 1st equality with either pos/neg iter coeff at last_var
            for (k = done; k < (int)dom.get_num_equalities(); ++k) {
                if (nfm_poly_coeff_is_pos(ctx_dom, dom.eqs_[k][last_var]) ||
                    nfm_poly_coeff_is_neg(ctx_dom, dom.eqs_[k][last_var])) {
                    break;
                }
            }
            if (k < (int)dom.get_num_equalities()) {
                break;
            }
        }
        if (last_var < 0) {
            //printf("  Can't find a valid equality to use as pivot\n");
            break;
        }
        if (k != done) { // We use k as pivot; move k to done
            //printf("Swapping %d with %d\n", k, done);
            nfm_domain_swap_equality(dom, k, done);
        }
        if (nfm_poly_coeff_is_neg(ctx_dom, dom.eqs_[done][last_var])) {
            dom.eqs_[done] = -dom.eqs_[done];
        }
        /*printf("Trying to eliminate var using equality (var: %d, done: %d): %s\n",
                last_var, done, dom.eqs_[done].to_string().c_str());*/
        bool is_feasible = nfm_domain_eliminate_var_using_equality(
            dom, last_var, dom.eqs_[done], progress);
        if (!is_feasible) {
            //printf("  Gaussian: infeasible domain\n");
            return NfmDomain::empty_domain(dom.get_coeff_space(), dom.get_space());
        }
    }
    return dom;
}

// Eliminate dim at index 'pos'. The sign of every coeff in the constraints at
// index 'pos' can't be unknown.
NfmDomain NfmSolver::nfm_domain_eliminate_dims_helper(
        NfmDomain& p_dom, size_t pos) {
    assert(pos < p_dom.space_.size());
    if (p_dom.is_empty() || p_dom.is_universe()) {
        //printf("  Domain is empty or it is a universe\n");
        return p_dom;
    }

    NfmDomain dom(p_dom);
    NfmContextDomain& ctx_dom = dom.get_context_domain();
    int i = 0, j = 0;
    for (i = 0; i < (int)dom.get_num_equalities(); ++i) {
        NfmConstraint& eq = dom.get_equality(i);
        NfmPolyCoeff& coeff = eq[pos];
        assert(!nfm_poly_coeff_is_unknown(ctx_dom, coeff));
        if (nfm_poly_coeff_is_zero(ctx_dom, coeff)) {
            continue;
        }
        bool is_feasible = nfm_domain_eliminate_var_using_equality(dom, pos, eq, NULL);
        if (!is_feasible) {
            //printf("  Contradiction in an (in-)equality, domain infeasible\n");
            return NfmDomain::empty_domain(dom.get_coeff_space(), dom.get_space());
        }
        nfm_domain_drop_equality(dom ,i);
        break;
    }
    if (i < (int)dom.get_num_equalities()) {
        return dom;
    }
    int n_lower = 0;
    int n_upper = 0;
    for (i = 0; i < (int)dom.get_num_inequalities(); ++i) {
        NfmConstraint& ineq = dom.get_inequality(i);
        NfmPolyCoeff& coeff = ineq[pos];
        assert(!nfm_poly_coeff_is_unknown(ctx_dom, coeff));
        if (nfm_poly_coeff_is_pos(ctx_dom, coeff)) {
            n_lower++;
        }
        else if (nfm_poly_coeff_is_neg(ctx_dom, coeff)) {
            n_upper++;
        }
    }
    for (i = dom.get_num_inequalities()-1; i >= 0; --i) {
        NfmConstraint& ineq_i = dom.get_inequality(i);
        NfmPolyCoeff& coeff_i = ineq_i[pos];
        if (nfm_poly_coeff_is_zero(ctx_dom, coeff_i)) {
            continue;
        }
        int last = -1;
        for (j = 0; j < i; ++j) {
            NfmConstraint& ineq_j = dom.get_inequality(j);
            NfmPolyCoeff& coeff_j = ineq_j[pos];
            if (nfm_poly_coeff_is_zero(ctx_dom, coeff_j)) {
                continue;
            }
            last = j;
            if (nfm_poly_coeff_get_sign(ctx_dom, coeff_i) ==
                    nfm_poly_coeff_get_sign(ctx_dom, coeff_j)) {
                continue;
            }
            NfmConstraint cst = nfm_constraint_elim(ctx_dom, dom.ineqs_[i], dom.ineqs_[j], pos);
            /*printf("  Eliminate %s with %s\n    Result: %s\n",
                   dom.ineqs_[i].to_string_with_sign().c_str(),
                   dom.ineqs_[j].to_string_with_sign().c_str(),
                   cst.to_string_with_sign().c_str());*/
            dom.add_constraint(std::move(cst));
        }
        nfm_domain_drop_inequality(dom, i);
        i = last + 1;
    }
    if (n_lower > 0 && n_upper > 0) {
        dom = nfm_domain_normalize_constraints(dom);
        dom = nfm_domain_remove_duplicate_constraints(dom, NULL);
        dom = nfm_domain_gauss(dom, NULL);
        if (dom.is_empty()) {
            //printf("  Empty domain\n");
            return dom;
        }
    }
    //printf("  RESULT dom:\n    %s\n", dom.to_string().c_str());
    //printf("    Context: %s\n", dom.get_context_domain().to_string().c_str());
    //printf("\n");
    return dom;
}

NfmUnionDomain NfmSolver::nfm_domain_eliminate_dims(
        const NfmDomain& p_dom, size_t pos) {
    if (pos >= p_dom.space_.size()) {
        return NfmUnionDomain(p_dom.coeff_space_, p_dom.space_);
    }
    size_t n = p_dom.space_.size() - pos;
    //printf("  Eliminating pos %d n %d", (int)pos, (int)n);
    return nfm_domain_eliminate_dims(p_dom, pos, n);
}

NfmUnionDomain NfmSolver::nfm_domain_eliminate_dims(const NfmDomain& p_dom,
        size_t pos, size_t n) {
    NfmUnionDomain union_dom(p_dom.coeff_space_, p_dom.space_);
    if ((n == 0) || (pos + n > p_dom.space_.size())) {
        return union_dom;
    }
    assert(pos + n == p_dom.space_.size());
    if (p_dom.is_empty() || p_dom.is_universe()) {
        return union_dom;
    }
    if (n == 0) {
        union_dom.add_domain(p_dom);
        return union_dom;
    }

    vector<NfmDomain> doms = {p_dom};
    // Eliminate vars from right to left, e.g. if dims are [x,y,z] then eliminate
    // from z to x in that order
    for (int d = pos+n-1; d >= 0 && d >= (int)pos; --d) {
        //printf("Elimination of %s (pos %d)\n", p_dom.space_[d].c_str(), d);
        vector<NfmDomain> temp;
        for (NfmDomain& dom : doms) {
            vector<NfmDomain> classified_doms =
                nfm_domain_classify_unknown_coefficient_helper(dom, d);
            for (NfmDomain& cdom : classified_doms) {
                //printf("From domain %s\n", cdom.to_string_with_sign().c_str());
                NfmDomain res_dom = nfm_domain_eliminate_dims_helper(cdom, (size_t)d);
                if (res_dom.is_empty()) {
                    // Since NfmUnionDomain is OR of NfmDomain, we can drop the
                    // NfmDomain if it's empty
                    //printf("  Empty domain; don't add to union_dom\n");
                    continue;
                }
                temp.push_back(std::move(res_dom));
            }
        }
        doms = temp;
    }
    for (size_t i = 0; i < doms.size(); ++i) {
        assert(!doms[i].is_empty());
        union_dom.add_domain(std::move(doms[i]));
    }
    return union_dom;
}

// Merge two union domains. Move all domains in union_dom2 to union_dom1.
// After call to this method, dom reference in union_dom2 is no longer valid.
void NfmSolver::nfm_union_domain_union(NfmUnionDomain& union_dom1,
                                       const NfmUnionDomain& union_dom2) {
    if (union_dom1.is_universe() || union_dom2.is_universe()) {
        // No need to merge if it's already a universe union_dom (any value is fine)
        NfmDomain universe_dom(union_dom1.coeff_space_, union_dom1.space_);
        union_dom1.domains_ = {std::move(universe_dom)};
        return;
    }

    for (size_t i = 0; i < union_dom2.get_num_domains(); ++i) {
        if (union_dom2.is_empty()) {
            continue;
        }
        union_dom1.add_domain(std::move(union_dom2[i]));
    }
}

NfmUnionDomain NfmSolver::nfm_union_domain_eliminate_dims(
        const NfmUnionDomain& p_union_dom, size_t pos) {
    if (pos >= p_union_dom.space_.size()) {
        return p_union_dom;
    }
    size_t n = p_union_dom.space_.size() - pos;
    return nfm_union_domain_eliminate_dims(p_union_dom, pos, n);
}

NfmUnionDomain NfmSolver::nfm_union_domain_eliminate_dims(
        const NfmUnionDomain& p_union_dom, size_t first, size_t n) {
    if ((n == 0) || (first + n > p_union_dom.space_.size())) {
        return p_union_dom;
    }
    NfmUnionDomain union_dom(p_union_dom.coeff_space_, p_union_dom.space_);
    for (size_t i = 0; i < p_union_dom.get_num_domains(); ++i) {
        //printf("\nEliminate vars domain %d\n", (int)i);
        NfmUnionDomain udom = nfm_domain_eliminate_dims(p_union_dom.domains_[i], first, n);
        nfm_union_domain_union(union_dom, udom);
    }
    return union_dom;
}

NfmDomain NfmSolver::nfm_domain_simplify(const NfmDomain& p_dom) {
    int progress = 1;
    if (p_dom.is_empty() || p_dom.is_universe()) {
        return p_dom;
    }
    NfmUnionDomain elim_dom = nfm_domain_eliminate_dims(p_dom, 0);
    if (elim_dom.is_empty()) {
        return NfmDomain::empty_domain(p_dom.coeff_space_, p_dom.space_);
    }
    NfmDomain dom(p_dom);
    while (progress) {
        progress = 0;
        dom = nfm_domain_normalize_constraints(dom);
        dom = nfm_domain_gauss(dom, &progress);
        dom = nfm_domain_remove_duplicate_constraints(dom, &progress);
    }
    /*printf("\nBefore simplify dom:\n    %s\n", p_dom.to_string().c_str());
    printf("After simplify dom:\n    %s\n\n", dom.to_string().c_str());*/
    return dom;
}

NfmUnionDomain NfmSolver::nfm_union_domain_simplify(const NfmUnionDomain& p_union_dom) {
    NfmUnionDomain union_dom(p_union_dom.coeff_space_, p_union_dom.space_);
    for (size_t i = 0; i < p_union_dom.get_num_domains(); ++i) {
        //printf("\nSimplify domain %d\n", (int)i);
        NfmDomain dom = nfm_domain_simplify(p_union_dom.domains_[i]);
        if (dom.is_empty()) { // (A || empty) = A
            continue;
        }
        if (dom.is_universe()) { // (A || universe) = universe
            union_dom.domains_ = {std::move(dom)};
            return union_dom;
        }
        union_dom.add_domain(dom);
    }
    return union_dom;
}

}
}
