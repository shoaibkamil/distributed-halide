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

#include "nfm_context_domain.h"

#include "nfm.h"
#include "nfm_debug.h"
#include "nfm_isl_interface.h"

namespace Nfm {
namespace Internal {

using std::ostream;
using std::ostringstream;
using std::string;
using std::vector;

namespace {

isl_basic_set* isl_basic_set_from_nfm_context_domain_linear(const NfmContextDomain& domain) {
    ostringstream stream;
    stream << "{[";
    const NfmSpace& space = domain.get_space();
    stream << space.to_string();
    stream << "] : " << domain.to_string_linear() << "}";

    /*printf(" isl_set_from_nfm_context_domain linear contexts only: %s\n",
        stream.str().c_str());*/

    struct isl_ctx *ctx = isl_ctx_alloc();
    assert(ctx != NULL);
    isl_basic_set *bset = isl_basic_set_read_from_str(ctx, stream.str().c_str());
    assert(bset != NULL);
    return bset;
}

vector<NfmContext> nfm_context_domain_from_bset(const NfmSpace& space, isl_basic_set *bset) {
    vector<NfmContext> contexts;

    isl_constraint_list *constraints_list = isl_basic_set_get_constraint_list(bset);

    for (int i = 0; i < isl_constraint_list_n_constraint(constraints_list); ++i) {
        NfmPolyCoeff polynom(space);
        isl_constraint *constraint_isl = isl_constraint_list_get_constraint(constraints_list, i);
        isl_aff *aff = isl_constraint_get_aff(constraint_isl);

        // Extract constant factor
        isl_val *cst_val = isl_aff_get_constant_val(aff);
        assert(cst_val != NULL);
        int cst_int = atoi(isl_val_to_str(cst_val));
        polynom = polynom.add(cst_int);
        isl_val_free(cst_val);

        size_t n_dims = space.size();
        for (size_t j = 0; j < n_dims; ++j) {
            isl_val *dim_val = isl_aff_get_coefficient_val(aff, isl_dim_in, j);
            int dim_int = atoi(isl_val_to_str(dim_val));

            vector<int> p_exp(space.size(), 0);
            p_exp[j] = 1;
            polynom = polynom.add(dim_int, p_exp, NFM_UNKNOWN);

            isl_val_free(dim_val);
        }

        bool eq = isl_constraint_is_equality(constraint_isl);
        NfmContext context(polynom, eq);
        contexts.push_back(std::move(context));

        isl_constraint_free(constraint_isl);
        isl_aff_free(aff);
    }

    isl_constraint_list_free(constraints_list);
    return contexts;
}

}

/********************************* NfmContextDomain *********************************/

string NfmContextDomain::to_string() const {
    ostringstream stream;
    if (is_empty()) {
        return "0 >= 1"; // Infeasible set
    }
    for (size_t i = 0; i < linear_.size(); ++i) {
        stream << linear_[i].to_string();
        if (i != linear_.size() - 1) {
            stream << " and ";
        }
    }
    if ((linear_.size() > 0) && (non_linear_.size() > 0)) {
        stream << " and ";
    }
    for (size_t i = 0; i < non_linear_.size(); ++i) {
        stream << non_linear_[i].to_string();
        if (i != non_linear_.size() - 1) {
            stream << " and ";
        }
    }
    return stream.str();
}

string NfmContextDomain::to_string_linear() const {
    ostringstream stream;
    if (is_empty()) {
        return "0 >= 1"; // Infeasible set
    }
    for (size_t i = 0; i < linear_.size(); ++i) {
        stream << linear_[i].to_string();
        if (i != linear_.size() - 1) {
            stream << " and ";
        }
    }
    return stream.str();
}

string NfmContextDomain::to_string_non_linear() const {
    ostringstream stream;
    if (is_empty()) {
        return "0 >= 1"; // Infeasible set
    }
    for (size_t i = 0; i < non_linear_.size(); ++i) {
        stream << non_linear_[i].to_string();
        if (i != non_linear_.size() - 1) {
            stream << " and ";
        }
    }
    return stream.str();
}


ostream& operator<<(ostream& out, const NfmContextDomain& dom) {
    return out << dom.to_string();
}

void NfmContextDomain::add_context(const NfmContext& context) {
    //std::cout << "ADDING CONTEXT: " << context.to_string_with_sign() << "\n";
    assert(context.get_space() == space_);
    if (context.is_linear()) {
        for (auto& ctx : linear_) {
            //std::cout << "  linear: " << ctx.to_string_with_sign() << "; equal? " << (ctx == context) << "\n";
            if (ctx == context) {
                //std::cout << "\t\tADDING DUPLICATE CONTEXT: " << context.to_string_with_sign() << "\n";
                return;
            }
        }
        linear_.push_back(context);
    } else {
        for (auto& ctx : non_linear_) {
            //std::cout << "  non-linear: " << ctx.to_string_with_sign() << "; equal? " << (ctx == context) << "\n";
            if (ctx == context) {
                //std::cout << "\t\tADDING DUPLICATE CONTEXT: " << context.to_string_with_sign() << "\n";
                return;
            }
        }
        non_linear_.push_back(context);
    }
}

void NfmContextDomain::add_context(const NfmContext&& context) {
    //std::cout << "ADDING CONTEXT: " << context.to_string_with_sign() << "\n";
    assert(context.get_space() == space_);
    if (context.is_linear()) {
        for (auto& ctx : linear_) {
            //std::cout << "   linear: " << ctx.to_string_with_sign() << "; equal? " << (ctx == context) << "\n";
            if (ctx == context) {
                //std::cout << "\t\tADDING DUPLICATE CONTEXT: " << context.to_string_with_sign() << "\n";
                return;
            }
        }
        linear_.push_back(std::move(context));
    } else {
        for (auto& ctx : non_linear_) {
            //std::cout << "   non-linear: " << ctx.to_string_with_sign() << "; equal? " << (ctx == context) << "\n";
            if (ctx == context) {
                //std::cout << "\t\tADDING DUPLICATE CONTEXT: " << context.to_string_with_sign() << "\n";
                return;
            }
        }
        non_linear_.push_back(std::move(context));
    }
}

// Compute the intersection of all the constraints in domain.
void NfmContextDomain::simplify() {
    IF_DEBUG(printf(" Starting simplify function for NfmContextDomain.\n"));
    IF_DEBUG(printf(" Tranforming from nfm_context_domain into isl_set.\n"));

    NfmSpace old_space = get_space();

    vector<string> dim_names = space_.get_names();
    for (size_t i = 0; i < dim_names.size(); ++i) {
        std::replace(dim_names[i].begin(), dim_names[i].end(), '$', '_');
        std::replace(dim_names[i].begin(), dim_names[i].end(), '.', '_');
        std::replace(dim_names[i].begin(), dim_names[i].end(), ':', '_');
        std::replace(dim_names[i].begin(), dim_names[i].end(), '[', '_');
        std::replace(dim_names[i].begin(), dim_names[i].end(), ']', '_');
    }

    update_space(NfmSpace(dim_names));

    isl_basic_set *bset = isl_basic_set_from_nfm_context_domain_linear(*this);
    assert(bset != NULL);
    if (isl_basic_set_is_empty(bset)) {
        *this = NfmContextDomain::empty_domain(space_);
    }

    linear_ = nfm_context_domain_from_bset(space_, bset);
    for (auto& ctx : non_linear_) {
        ctx.simplify();
    }
    isl_basic_set_free(bset);

    update_space(old_space);
}

void NfmContextDomain::sort() {
    std::sort(linear_.begin(), linear_.end());
    std::sort(non_linear_.begin(), non_linear_.end());
}

bool operator==(const NfmContextDomain& lhs, const NfmContextDomain& rhs) {
    if (lhs.flag_ != rhs.flag_) {
        return false;
    }
    if (lhs.linear_.size() != rhs.linear_.size()) {
        return false;
    }
    if (lhs.non_linear_.size() != rhs.non_linear_.size()) {
        return false;
    }
    if (lhs.space_ != rhs.space_) {
        return false;
    }

    if (!std::is_permutation(lhs.linear_.begin(), lhs.linear_.end(), rhs.linear_.begin())) {
        return false;
    }
    if (!std::is_permutation(lhs.non_linear_.begin(), lhs.non_linear_.end(), rhs.non_linear_.begin())) {
        return false;
    }
    return true;
}

bool operator!=(const NfmContextDomain& lhs, const NfmContextDomain& rhs) {
    return !(lhs == rhs);
}

}
}
