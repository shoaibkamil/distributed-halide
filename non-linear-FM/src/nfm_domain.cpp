#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>

#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/val.h>

#include "nfm_domain.h"
#include "nfm.h"
#include "nfm_isl_interface.h"

#include "nfm_debug.h"

namespace Nfm {
namespace Internal {

using std::ostream;
using std::ostringstream;
using std::string;
using std::vector;

namespace {

isl_set* isl_set_from_nfm_domain(const NfmDomain& domain, struct isl_ctx* ctx) {
    ostringstream stream;
    stream << "[";
    const NfmSpace& coeff_space_ = domain.get_coeff_space();
    stream << coeff_space_.to_string();
    stream << "] -> {[";

    const NfmSpace& space = domain.get_space();
    stream << space.to_string();
    stream << "] : " << domain.to_string() << "}";

    //printf(" isl_set_from_nfm_domain: %s\n", stream.str().c_str());

    assert(ctx != NULL);
    isl_set *set = isl_set_read_from_str(ctx, stream.str().c_str());
    assert(set != NULL);
    return set;
}

NfmDomain nfm_domain_from_bset(const NfmSpace& coeff_space, const NfmSpace& space,
                               isl_basic_set *bset) {
    if (isl_basic_set_is_empty(bset)) {
        return NfmDomain::empty_domain(coeff_space, space);
    }

    NfmDomain domain(coeff_space, space);

    isl_constraint_list *constraints_list = isl_basic_set_get_constraint_list(bset);

    for (int i = 0; i < isl_constraint_list_n_constraint(constraints_list); ++i) {
        NfmPoly polynom(coeff_space, space);
        isl_constraint *constraint_isl = isl_constraint_list_get_constraint(constraints_list, i);
        isl_aff *aff = isl_constraint_get_aff(constraint_isl);

        // Extract constant factor
        isl_val *cst_val = isl_aff_get_constant_val(aff);
        assert(cst_val != NULL);
        int cst_int = atoi(isl_val_to_str(cst_val));
        polynom = polynom.add(cst_int);
        isl_val_free(cst_val);

        // Extract constant (symbolic)
        size_t n_params = coeff_space.size();
        for (size_t j = 0; j < n_params; ++j) {
            isl_val *param_val = isl_aff_get_coefficient_val(aff, isl_dim_param, j);
            int param_int = atoi(isl_val_to_str(param_val));

            vector<int> param_exp(coeff_space.size(), 0);
            param_exp[j] = 1;
            polynom = polynom.add(param_int, param_exp, NFM_UNKNOWN);

            isl_val_free(param_val);
        }

        size_t n_dims = space.size();
        for (size_t j = 0; j < n_dims; ++j) {
            isl_val *dim_val = isl_aff_get_coefficient_val(aff, isl_dim_in, j);
            int dim_int = atoi(isl_val_to_str(dim_val));

            vector<int> exp_inner(coeff_space.size(), 0);
            vector<int> exp_outer(space.size(), 0);
            exp_outer[j] = 1;
            polynom = polynom.add(dim_int, exp_inner, NFM_UNKNOWN, exp_outer);

            isl_val_free(dim_val);
        }

        bool eq = isl_constraint_is_equality(constraint_isl);
        NfmConstraint constraint(polynom, eq);
        domain.add_constraint(constraint);

        isl_constraint_free(constraint_isl);
        isl_aff_free(aff);
    }

    isl_constraint_list_free(constraints_list);
    return domain;
}

// Compute the union of all the constraints in union_dom
isl_bset_list *isl_bset_list_from_nfm_union_domain(
        const NfmUnionDomain& union_dom, struct isl_ctx *ctx) {
    assert(ctx != NULL);

    IF_DEBUG(printf(" Starting the union function.\n"));

    isl_set *set = NULL;
    for (const auto& domain : union_dom.get_domains()) {
        isl_set *other_set = isl_set_from_nfm_domain(domain, ctx);
        if (set == NULL) {
            set = other_set;
        } else {
            set = isl_set_union(set, other_set);
        }
    }
    IF_DEBUG(printf(" End of the union function.\n"));
    assert(set != NULL);
    isl_bset_list *result = isl_set_get_bsets_list(set);

    isl_set_free(set);
    return result;
}

NfmUnionDomain nfm_union_domain_from_isl_bset_list(
        const NfmSpace& p_coeff_space,
        const NfmSpace& p_space,
        isl_bset_list *bset_list) {
    NfmUnionDomain result(p_coeff_space, p_space);
    isl_bset_list *node = bset_list;

    while (node != NULL) {
        NfmDomain domain = nfm_domain_from_bset(p_coeff_space, p_space, node->bset);
        if (domain.is_empty()) {
            continue;
        }
        result.add_domain(domain);
        node = node->next;
    }
    return result;
}

}

/********************************* NfmDomain *********************************/

string NfmDomain::to_string() const {
    ostringstream stream;
    if (is_empty()) {
        return "0 >= 1"; // Infeasible set
    }
    for (size_t i = 0; i < eqs_.size(); ++i) {
        stream << eqs_[i].to_string();
        if (i != eqs_.size() - 1) {
            stream << " and ";
        }
    }
    if ((eqs_.size() > 0) && (ineqs_.size() > 0)) {
        stream << " and ";
    }
    for (size_t i = 0; i < ineqs_.size(); ++i) {
        stream << ineqs_[i].to_string();
        if (i != ineqs_.size() - 1) {
            stream << " and ";
        }
    }
    return stream.str();
}

string NfmDomain::to_string_with_sign() const {
    ostringstream stream;
    if (is_empty()) {
        return "0 >= 1"; // Infeasible set
    }
    for (size_t i = 0; i < eqs_.size(); ++i) {
        stream << "(" << eqs_[i].to_string_with_sign() << ")";
        if (i != eqs_.size() - 1) {
            stream << " and ";
        }
    }
    if ((eqs_.size() > 0) && (ineqs_.size() > 0)) {
        stream << " and ";
    }
    for (size_t i = 0; i < ineqs_.size(); ++i) {
        stream << "(" << ineqs_[i].to_string_with_sign() << ")";
        if (i != ineqs_.size() - 1) {
            stream << " and ";
        }
    }
    return stream.str();
}

ostream& operator<<(ostream& out, const NfmDomain& dom) {
    return out << dom.to_string();
}

void NfmDomain::add_constraint(const NfmConstraint& constraint) {
    assert(constraint.get_space() == space_);
    assert(constraint.get_coeff_space() == coeff_space_);
    if (constraint.is_equality()) {
        eqs_.push_back(constraint);
    } else {
        ineqs_.push_back(constraint);
    }
}

void NfmDomain::add_constraint(const NfmConstraint&& constraint) {
    assert(constraint.get_space() == space_);
    assert(constraint.get_coeff_space() == coeff_space_);
    if (constraint.is_equality()) {
        eqs_.push_back(std::move(constraint));
    } else {
        ineqs_.push_back(std::move(constraint));
    }
}

void NfmDomain::add_context(const NfmContext& context) {
    assert(context.get_space() == context_dom_.get_space());
    context_dom_.add_context(context);
}

void NfmDomain::add_context(const NfmContext&& context) {
    assert(context.get_space() == context_dom_.get_space());
    context_dom_.add_context(context);
}

// Compute the intersection of all the constraints in domain
NfmUnionDomain NfmDomain::simplify() {
    IF_DEBUG(printf(" Starting intersection function.\n"));
    IF_DEBUG(printf(" Tranforming from nfm_domain into isl_set.\n"));

    struct isl_ctx *ctx = isl_ctx_alloc();
    isl_set *set = isl_set_from_nfm_domain(*this, ctx);
    assert(set != NULL);

    /*printf("\n");
    isl_set_dump(set);
    printf("\n");*/

    isl_bset_list *bset_list = isl_set_get_bsets_list(set);
    assert(bset_list != NULL);
    NfmUnionDomain union_dom = nfm_union_domain_from_isl_bset_list(
        coeff_space_, space_, bset_list);

    isl_set_free(set);
    isl_bset_list_free(bset_list);
    isl_ctx_free(ctx);

    return union_dom;
}

void NfmDomain::sort() {
    context_dom_.sort();
    std::sort(eqs_.begin(), eqs_.end());
    std::sort(ineqs_.begin(), ineqs_.end());
}

// Order by domain with least number of eqs. If have equal number of eqs, ordered
// by number of ineqs (less first)
bool NfmDomain::operator<(const NfmDomain& other) const {
    assert(coeff_space_ == other.coeff_space_);
    assert(space_ == other.space_);

    if (eqs_.size() < other.eqs_.size()) {
        return true;
    } else if (eqs_.size() > other.eqs_.size()) {
        return false;
    }

    if (ineqs_.size() < other.ineqs_.size()) {
        return true;
    } else if (ineqs_.size() > other.ineqs_.size()) {
        return false;
    }

    assert((eqs_.size() == other.eqs_.size()) && (ineqs_.size() == other.ineqs_.size()));

    for (size_t i = 0; i < eqs_.size() && i < other.eqs_.size(); ++i) {
        if (eqs_[i] == other.eqs_[i]) {
            continue;
        } else if (eqs_[i] < other.eqs_[i]) {
            return true;
        } else { // eqs_[i] > other.eqs_[i])
            return false;
        }
    }
    for (size_t i = 0; i < ineqs_.size() && i < other.ineqs_.size(); ++i) {
        if (ineqs_[i] == other.ineqs_[i]) {
            continue;
        } else if (ineqs_[i] < other.ineqs_[i]) {
            return true;
        } else { // ineqs_[i] > other.ineqs_[i])
            return false;
        }
    }
    return true;
}

bool operator==(const NfmDomain& lhs, const NfmDomain& rhs) {
    if (lhs.flag_ != rhs.flag_) {
        return false;
    }
    if (lhs.eqs_.size() != rhs.eqs_.size()) {
        return false;
    }
    if (lhs.ineqs_.size() != rhs.ineqs_.size()) {
        return false;
    }
    if (lhs.coeff_space_ != rhs.coeff_space_) {
        return false;
    }
    if (lhs.space_ != rhs.space_) {
        return false;
    }
    if (lhs.context_dom_ != rhs.context_dom_) {
        return false;
    }
    if (!std::is_permutation(lhs.eqs_.begin(), lhs.eqs_.end(), rhs.eqs_.begin())) {
        return false;
    }
    if (!std::is_permutation(lhs.ineqs_.begin(), lhs.ineqs_.end(), rhs.ineqs_.begin())) {
        return false;
    }
    return true;
}

bool operator!=(const NfmDomain& lhs, const NfmDomain& rhs) {
    return !(lhs == rhs);
}


/******************************* NfmUnionDomain *******************************/

string NfmUnionDomain::to_string() const {
    if (is_empty()) {
        return "0 >= 1"; // Infeasible set
    }
    ostringstream stream;
    for (size_t i = 0; i < domains_.size(); ++i) {
        if (domains_[i].is_empty()) {
            continue;
        }
        stream << "(" << domains_[i].to_string() << ")";
        if (i != domains_.size() - 1) {
            stream << " or ";
        }
    }
    return stream.str();
}

ostream& operator<<(ostream& out, const NfmUnionDomain& union_dom) {
    return out << union_dom.to_string();
}

void NfmUnionDomain::add_domain(const NfmDomain& domain) {
    if (domain.is_empty()) { // (A || empty) == A
        return;
    } else if (domain.is_universe()) { // (A || universe) == universe
        domains_ = {domain};
    } else {
        for (auto& dom : domains_) {
            if (dom == domain) {
                //std::cout << "\t\tADDING DUPLICATE DOMAIN: " << domain << "\n";
                return;
            }
        }
        domains_.push_back(domain);
    }
}

void NfmUnionDomain::add_domain(const NfmDomain&& domain) {
    if (domain.is_empty()) { // (A || empty) == A
        return;
    } else if (domain.is_universe()) { // (A || universe) == universe
        domains_ = {std::move(domain)};
    } else {
        for (auto& dom : domains_) {
            if (dom == domain) {
                //std::cout << "\t\tADDING DUPLICATE DOMAIN: " << domain << "\n";
                return;
            }
        }
        domains_.push_back(std::move(domain));
    }
}

// Return true if there is no domain in the union domain or
// if all the domains within the union domain are empty
bool NfmUnionDomain::is_empty() const {
    for (const auto& dom : domains_) {
        if (!dom.is_empty()) {
            return false;
        }
    }
    return true;
}

bool NfmUnionDomain::is_universe() const {
    for (const auto& dom : domains_) {
        if (dom.is_universe()) {
            return true;
        }
    }
    return false;
}

// Compute the union of all the domains_ in a union_domain
NfmUnionDomain NfmUnionDomain::simplify() {
    IF_DEBUG(fprintf(stdout, " Starting the union function.\n"));

    NfmSpace old_coeff_space = get_coeff_space();;
    NfmSpace old_space = get_space();

    vector<string> coeff_names = coeff_space_.get_names();
    for (size_t i = 0; i < coeff_names.size(); ++i) {
        std::replace(coeff_names[i].begin(), coeff_names[i].end(), '$', '_');
        std::replace(coeff_names[i].begin(), coeff_names[i].end(), '.', '_');
        std::replace(coeff_names[i].begin(), coeff_names[i].end(), ':', '_');
    }

    vector<string> dim_names = space_.get_names();
    for (size_t i = 0; i < dim_names.size(); ++i) {
        std::replace(dim_names[i].begin(), dim_names[i].end(), '$', '_');
        std::replace(dim_names[i].begin(), dim_names[i].end(), '.', '_');
        std::replace(dim_names[i].begin(), dim_names[i].end(), ':', '_');
    }

    update_coeff_space(NfmSpace(coeff_names));
    update_space(NfmSpace(dim_names));

    struct isl_ctx *ctx = isl_ctx_alloc();
    isl_bset_list *bset_list = isl_bset_list_from_nfm_union_domain(*this, ctx);
    assert(bset_list != NULL);

    NfmUnionDomain union_dom = nfm_union_domain_from_isl_bset_list(
        coeff_space_, space_, bset_list);

    isl_bset_list_free(bset_list);
    isl_ctx_free(ctx);

    union_dom.update_coeff_space(old_coeff_space);
    union_dom.update_space(old_space);
    update_coeff_space(old_coeff_space);
    update_space(old_space);

    return union_dom;
}

void NfmUnionDomain::sort() {
    for (auto& dom : domains_) {
        dom.sort();
    }
    std::sort(domains_.begin(), domains_.end());
}

bool operator==(const NfmUnionDomain& lhs, const NfmUnionDomain& rhs) {
    if (lhs.domains_.size() != rhs.domains_.size()) {
        return false;
    }
    if (lhs.coeff_space_ != rhs.coeff_space_) {
        return false;
    }
    if (lhs.space_ != rhs.space_) {
        return false;
    }
    if (!std::is_permutation(lhs.domains_.begin(), lhs.domains_.end(), rhs.domains_.begin())) {
        return false;
    }
    return true;
}

bool operator!=(const NfmUnionDomain& lhs, const NfmUnionDomain& rhs) {
    return !(lhs == rhs);
}

}
}
