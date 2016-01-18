#ifndef NFM_SOLVER_H
#define NFM_SOLVER_H

#include <assert.h>
#include <string>
#include <vector>

#include "nfm_constraint.h"
#include "nfm_context.h"
#include "nfm_context_domain.h"
#include "nfm_domain.h"
#include "nfm_polynom.h"
#include "nfm_polynom_frac.h"
#include "nfm_space.h"

namespace Nfm {
namespace Internal {

class NfmSolver {
public:
    std::string to_string() const;

    static bool nfm_constraint_ineq_is_redundant(
        const NfmDomain& p_dom, const NfmConstraint& cst);
    static bool nfm_constraint_ineq_is_redundant(
        const NfmUnionDomain& p_union_dom, const NfmConstraint& cst);

    //TODO: implement this
    static NfmUnionDomain nfm_union_domain_make_disjoint(
        const NfmUnionDomain& p_union_dom);

    static NfmUnionDomain nfm_union_domain_simplify(
        const NfmUnionDomain& p_union_dom);
    static NfmDomain nfm_domain_simplify(const NfmDomain& p_dom);

    static NfmUnionDomain nfm_union_domain_eliminate_dims(
        const NfmUnionDomain& p_union_dom, size_t pos);

    static NfmUnionDomain nfm_union_domain_eliminate_dims(
        const NfmUnionDomain& p_union_dom, size_t first, size_t n);

    static NfmUnionDomain nfm_domain_eliminate_dims(
        const NfmDomain& p_dom, size_t pos);

    static NfmUnionDomain nfm_domain_eliminate_dims(
        const NfmDomain& p_dom, size_t pos, size_t n);

    static NfmUnionDomain nfm_union_domain_classify_unknown_coefficient(
        NfmUnionDomain& p_union_dom, size_t pos);
    static NfmUnionDomain nfm_domain_classify_unknown_coefficient(
        NfmDomain& p_dom, size_t pos);

    // Return the sign of a NfmPolyCoeff based on the context domain. This
    // function may modify the sign if it's originally unknown.
    static NfmSign nfm_poly_coeff_get_sign(const NfmContextDomain& ctx_dom,
        NfmPolyCoeff& coeff);

    static bool nfm_poly_coeff_is_pos(const NfmContextDomain& ctx_dom,
        NfmPolyCoeff& coeff);
    static bool nfm_poly_coeff_is_neg(const NfmContextDomain& ctx_dom,
        NfmPolyCoeff& coeff);
    static bool nfm_poly_coeff_is_zero(const NfmContextDomain& ctx_dom,
        NfmPolyCoeff& coeff);
    static bool nfm_poly_coeff_is_non_neg(const NfmContextDomain& ctx_dom,
        NfmPolyCoeff& coeff);
    static bool nfm_poly_coeff_is_non_pos(const NfmContextDomain& ctx_dom,
        NfmPolyCoeff& coeff);
    static bool nfm_poly_coeff_is_unknown(const NfmContextDomain& ctx_dom,
        NfmPolyCoeff& coeff);

    // Return the sign of a NfmPolyCoeff based on the context domain. This
    // function may modify the sign if it's originally unknown.
    static NfmSign nfm_poly_frac_get_sign(const NfmContextDomain& ctx_dom,
        NfmPolyFrac& frac);

    static bool nfm_poly_frac_is_pos(const NfmContextDomain& ctx_dom,
        NfmPolyFrac& frac);
    static bool nfm_poly_frac_is_neg(const NfmContextDomain& ctx_dom,
        NfmPolyFrac& frac);
    static bool nfm_poly_frac_is_zero(const NfmContextDomain& ctx_dom,
        NfmPolyFrac& frac);
    static bool nfm_poly_frac_is_non_neg(const NfmContextDomain& ctx_dom,
        NfmPolyFrac& frac);
    static bool nfm_poly_frac_is_non_pos(const NfmContextDomain& ctx_dom,
        NfmPolyFrac& frac);
    static bool nfm_poly_frac_is_unknown(const NfmContextDomain& ctx_dom,
        NfmPolyFrac& frac);

    static bool nfm_constraint_involves(const NfmContextDomain& ctx_dom,
        NfmConstraint& cst, const std::string& var);
    static bool nfm_constraint_involves(const NfmContextDomain& ctx_dom,
        NfmConstraint& cst, int var_idx);
    static bool nfm_constraint_is_upper_bound(const NfmContextDomain& ctx_dom,
        NfmConstraint& cst, const std::string& var);
    static bool nfm_constraint_is_upper_bound(const NfmContextDomain& ctx_dom,
        NfmConstraint& cst, int var_idx);
    static bool nfm_constraint_is_lower_bound(const NfmContextDomain& ctx_dom,
        NfmConstraint& cst, const std::string& var);
    static bool nfm_constraint_is_lower_bound(const NfmContextDomain& ctx_dom,
        NfmConstraint& cst, int var_idx);

private:
    struct NfmContextConstraint {
        NfmContext context;
        NfmConstraint constraint;
    };

    static NfmConstraint nfm_constraint_get_opposite(const NfmConstraint& cst);

    static std::vector<NfmContextConstraint> nfm_constraint_classify_unknown_coefficient(
        const NfmConstraint& p_cst, size_t pos);

    static void nfm_union_domain_union(NfmUnionDomain& union_dom1,
                                       const NfmUnionDomain& union_dom2);

    static int nfm_domain_drop_equality(NfmDomain& dom, size_t pos);
    static int nfm_domain_drop_inequality(NfmDomain& dom, size_t pos);

    static void nfm_domain_swap_equality(NfmDomain& dom, int a, int b);
    static void nfm_domain_swap_inequality(NfmDomain& dom, int a, int b);

    static void nfm_domain_inequality_to_equality(NfmDomain& dom,
        size_t pos1, size_t pos2);

    static bool nfm_domain_eliminate_var_using_equality(
        NfmDomain& dom, size_t pos, NfmConstraint& eq, int *progress);

    static NfmDomain nfm_domain_normalize_constraints(NfmDomain& dom);

    static NfmDomain nfm_domain_remove_duplicate_constraints(
        NfmDomain& dom, int *progress);

    static NfmDomain nfm_domain_gauss(NfmDomain& dom, int *progress);

    static NfmConstraint nfm_constraint_elim_helper(const NfmContextDomain& ctx_dom,
        const NfmConstraint& pos_cst, const NfmConstraint& neg_cst, size_t pos);
    static NfmConstraint nfm_constraint_elim(const NfmContextDomain& ctx_dom,
        NfmConstraint& dst, NfmConstraint& src, size_t pos);

    static NfmDomain nfm_domain_eliminate_dims_helper(
        NfmDomain& dom, size_t pos);

    // Return the sign of a linear polynomial based on the context domain
    static NfmSign nfm_poly_coeff_linear_get_sign(const NfmContextDomain& ctx_dom,
        const NfmPolyCoeff& coeff);
    // Return the sign of a non-linear polynomial based on the context domain
    static NfmSign nfm_poly_coeff_non_linear_get_sign(const NfmContextDomain& ctx_dom,
        const NfmPolyCoeff& coeff);

    static std::vector<NfmDomain> nfm_domain_classify_unknown_coefficient_helper(
        NfmDomain& p_dom, size_t pos);

};


}
}

#endif
