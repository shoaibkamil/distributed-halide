#ifndef HALIDE_NFM_TO_HALIDE_H
#define HALIDE_NFM_TO_HALIDE_H

#include <iostream>
#include <sstream>
#include <set>

#include <nfm_constraint.h>
#include <nfm_domain.h>
#include <nfm_polynom.h>

#include "Bounds.h"
#include "IREquality.h"

namespace Halide {
namespace Internal {

/*Box convert_nfm_union_domain_to_halide_box(
    Type type, const Nfm::Internal::NfmUnionDomain& p_union_dom,
    const std::vector<std::string>& box_dims,
    const std::vector<std::string> *let_assignments=NULL,
    const std::map<std::string, Expr> *expr_substitutions=NULL,
    std::vector<std::pair<std::string, Expr>> *let_substitutions=NULL);*/

Interval convert_nfm_union_domain_to_halide_interval(
    Type type, const Nfm::Internal::NfmUnionDomain& p_union_dom,
    const std::string& dim_name,
    const std::vector<std::string> *let_assignments=NULL,
    const std::map<std::string, Expr> *expr_substitutions=NULL,
    const std::vector<std::pair<std::string, Expr>> *let_substitutions=NULL);

Expr convert_nfm_union_domain_to_halide_expr(
    Type type, Nfm::Internal::NfmUnionDomain& union_dom,
    const std::vector<std::string> *let_assignments=NULL,
    const std::map<std::string, Expr> *expr_substitutions=NULL,
    const std::vector<std::pair<std::string, Expr>> *let_substitutions=NULL);

Expr nfm_simplify_expr(const Expr& expr);

Interval nfm_simplify_interval(const Interval& interval);

}
}

#endif
