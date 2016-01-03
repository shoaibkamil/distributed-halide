#ifndef NFM_DOMAIN_H
#define NFM_DOMAIN_H

#include <assert.h>
#include <string>
#include <vector>

#include "nfm_constraint.h"
#include "nfm_context.h"
#include "nfm_context_domain.h"
#include "nfm_space.h"

namespace Nfm {
namespace Internal {

class NfmSolver;
class NfmUnionDomain;

class NfmDomain {
public:
    friend class NfmSolver;

    enum NfmDomainFlag {
        NFM_DOMAIN_NONE,
        NFM_DOMAIN_EMPTY
    };

    NfmDomain(const NfmSpace& p_coeff_space,
              const NfmSpace& p_space,
              NfmDomainFlag flag,
              const NfmContextDomain& ctx_dom)
            : coeff_space_(p_coeff_space)
            , space_(p_space)
            , flag_(flag)
            , context_dom_(ctx_dom) {
        assert(p_coeff_space == ctx_dom.get_space());
    }

    NfmDomain(const std::vector<std::string>& coeff_names,
              const std::vector<std::string>& dim_names,
              NfmDomainFlag flag,
              const NfmContextDomain& ctx_dom)
        : NfmDomain(NfmSpace(coeff_names), NfmSpace(dim_names), flag, ctx_dom) {}

    NfmDomain(const NfmSpace& p_coeff_space,
              const NfmSpace& p_space,
              const NfmContextDomain& ctx_dom)
        : NfmDomain(p_coeff_space, p_space, NFM_DOMAIN_NONE, ctx_dom) {}

    NfmDomain(const std::vector<std::string>& coeff_names,
              const std::vector<std::string>& dim_names,
              const NfmContextDomain& ctx_dom)
        : NfmDomain(coeff_names, dim_names, NFM_DOMAIN_NONE, ctx_dom) {}

    NfmDomain(const NfmSpace& p_coeff_space,
              const NfmSpace& p_space,
              NfmDomainFlag flag)
        : NfmDomain(p_coeff_space, p_space, flag, NfmContextDomain(p_coeff_space)) {}

    NfmDomain(const std::vector<std::string>& coeff_names,
              const std::vector<std::string>& dim_names,
              NfmDomainFlag flag)
        : NfmDomain(NfmSpace(coeff_names), NfmSpace(dim_names), flag) {}

    NfmDomain(const NfmSpace& p_coeff_space,
              const NfmSpace& p_space)
        : NfmDomain(p_coeff_space, p_space, NFM_DOMAIN_NONE) {}

    NfmDomain(const std::vector<std::string>& coeff_names,
              const std::vector<std::string>& dim_names)
        : NfmDomain(coeff_names, dim_names, NFM_DOMAIN_NONE) {}

    static NfmDomain empty_domain(const NfmSpace& p_coeff_space,
                                  const NfmSpace& p_space) {
        return NfmDomain(p_coeff_space, p_space, NFM_DOMAIN_EMPTY);
    }

    std::string to_string() const;
    std::string to_string_with_sign() const;
    friend std::ostream& operator<<(std::ostream& out, const NfmDomain& dom);

    // Add constraint to a domain of quasi-polynomials.
    void add_constraint(const NfmConstraint& constraint);
    void add_constraint(const NfmConstraint&& constraint);

    // Add constraint to a domain of quasi-polynomials.
    void add_context(const NfmContext& context);
    void add_context(const NfmContext&& context);

    // Compute the intersection of all the constraints in domain.
    NfmUnionDomain simplify();

    void sort();

    bool is_empty() const { return (flag_ == NFM_DOMAIN_EMPTY); }
    bool is_universe() const { return (get_num_constraints() == 0); }

    const NfmSpace& get_space() const { return space_; }
    const NfmSpace& get_coeff_space() const { return coeff_space_; }

    const NfmContextDomain& get_context_domain() const { return context_dom_; }
    NfmContextDomain& get_context_domain() { return context_dom_; }

    size_t get_num_equalities() const { return eqs_.size(); }
    size_t get_num_inequalities() const { return ineqs_.size(); }
    size_t get_num_constraints() const {
        return (get_num_equalities() + get_num_inequalities());
    }

    const std::vector<NfmConstraint>& get_equalities() const { return eqs_; }
    std::vector<NfmConstraint>& get_equalities() { return eqs_; }
    const std::vector<NfmConstraint>& get_inequalities() const { return ineqs_; }
    std::vector<NfmConstraint>& get_inequalities() { return ineqs_; }

    std::vector<NfmConstraint> get_constraints() const {
        std::vector<NfmConstraint> result(eqs_);
        for (const auto& cst : ineqs_) {
            result.push_back(cst);
        }
        return result;
    }

    const NfmConstraint& get_equality(size_t idx) const {
        assert(idx < eqs_.size());
        return eqs_[idx];
    }
    NfmConstraint& get_equality(size_t idx) {
        assert(idx < eqs_.size());
        return eqs_[idx];
    }
    const NfmConstraint& get_inequality(size_t idx) const {
        assert(idx < ineqs_.size());
        return ineqs_[idx];
    }
    NfmConstraint& get_inequality(size_t idx) {
        assert(idx < ineqs_.size());
        return ineqs_[idx];
    }

    friend bool operator==(const NfmDomain& lhs, const NfmDomain& rhs);
    friend bool operator!=(const NfmDomain& lhs, const NfmDomain& rhs);

    // For comparison in map/set/etc
    bool operator<(const NfmDomain& other) const;

    /* Hack to make it work with isl */
    void update_coeff_space(const NfmSpace& p_coeff_space) {
        coeff_space_ = p_coeff_space;
        for (auto& cst : eqs_) {
            cst.update_coeff_space(p_coeff_space);
        }
        for (auto& cst : ineqs_) {
            cst.update_coeff_space(p_coeff_space);
        }
    }
    void update_space(const NfmSpace& p_space) {
        space_ = p_space;
        for (auto& cst : eqs_) {
            cst.update_space(p_space);
        }
        for (auto& cst : ineqs_) {
            cst.update_space(p_space);
        }
    }

private:
    NfmSpace coeff_space_;
    NfmSpace space_;
    std::vector<NfmConstraint> eqs_;
    std::vector<NfmConstraint> ineqs_;
    NfmDomainFlag flag_;
    NfmContextDomain context_dom_;
};

class NfmUnionDomain {
public:
    friend class NfmSolver;

    NfmUnionDomain(const NfmSpace& p_coeff_space, const NfmSpace& p_space,
                   const std::vector<NfmDomain>& doms)
        : coeff_space_(p_coeff_space)
        , space_(p_space)
        , domains_(doms) {}

    NfmUnionDomain(const NfmSpace& p_coeff_space, const NfmSpace& p_space)
        : coeff_space_(p_coeff_space)
        , space_(p_space) {}

    NfmUnionDomain(const std::vector<std::string>& coeff_names,
                   const std::vector<std::string>& dim_names)
        : NfmUnionDomain(NfmSpace(coeff_names), NfmSpace(dim_names)) {}

    std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& out,
                                    const NfmUnionDomain& union_dom);

    void add_domain(const NfmDomain& domain);
    void add_domain(const NfmDomain&& domain);

    // Compute the union of all the domains_ in a union_domain.
    NfmUnionDomain simplify();

    void sort();

    bool is_empty() const;
    bool is_universe() const;

    const NfmSpace& get_space() const { return space_; }
    const NfmSpace& get_coeff_space() const { return coeff_space_; }

    size_t get_num_domains() const { return domains_.size(); }

    const std::vector<NfmDomain>& get_domains() const { return domains_; }
    std::vector<NfmDomain>& get_domains() { return domains_; }

    const NfmDomain& get_domain(size_t idx) const { return domains_[idx]; }
    NfmDomain& get_domain(size_t idx) { return domains_[idx]; }
    const NfmDomain& operator[](size_t idx) const {
        const NfmDomain& dom = get_domain(idx);
        return dom;
    }
    NfmDomain& operator[](size_t idx) {
        NfmDomain& dom = get_domain(idx);
        return dom;
    }

    friend bool operator==(const NfmUnionDomain& lhs, const NfmUnionDomain& rhs);
    friend bool operator!=(const NfmUnionDomain& lhs, const NfmUnionDomain& rhs);

    /* Hack to make it work with isl */
    void update_coeff_space(const NfmSpace& p_coeff_space) {
        coeff_space_ = p_coeff_space;
        for (auto& dom : domains_) {
            dom.update_coeff_space(p_coeff_space);
        }
    }
    void update_space(const NfmSpace& p_space) {
        space_ = p_space;
        for (auto& dom : domains_) {
            dom.update_space(p_space);
        }
    }
private:
    NfmSpace coeff_space_;
    NfmSpace space_;
    std::vector<NfmDomain> domains_;
};

}
}

#endif
