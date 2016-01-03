#ifndef NFM_CONTEXT_DOMAIN_H
#define NFM_CONTEXT_DOMAIN_H

#include <assert.h>
#include <string>
#include <vector>

#include "nfm_context.h"
#include "nfm_space.h"

namespace Nfm {
namespace Internal {

class NfmSolver;

class NfmContextDomain {
public:
    friend class NfmSolver;

    enum NfmContextDomainFlag {
        NFM_CONTEXT_DOMAIN_NONE,
        NFM_CONTEXT_DOMAIN_EMPTY
    };

    NfmContextDomain(const NfmSpace& p_space, NfmContextDomainFlag flag)
        : space_(p_space)
        , flag_(flag) {}

    NfmContextDomain(const std::vector<std::string>& dim_names,
                     NfmContextDomainFlag flag)
        : NfmContextDomain(NfmSpace(dim_names), flag) {}

    NfmContextDomain(const NfmSpace& p_space)
        : NfmContextDomain(p_space, NFM_CONTEXT_DOMAIN_NONE) {}

    NfmContextDomain(const std::vector<std::string>& dim_names)
        : NfmContextDomain(dim_names, NFM_CONTEXT_DOMAIN_NONE) {}

    static NfmContextDomain empty_domain(const NfmSpace& p_space) {
        return NfmContextDomain(p_space, NFM_CONTEXT_DOMAIN_EMPTY);
    }

    std::string to_string() const;
    std::string to_string_linear() const;
    std::string to_string_non_linear() const;
    friend std::ostream& operator<<(std::ostream& out, const NfmContextDomain& dom);

    /* Add context to the domain of quasi-polynomials. */
    void add_context(const NfmContext& context);
    void add_context(const NfmContext&& context);

    /* Compute the intersection of all the constraints in domain. */
    void simplify();

    void sort();

    bool is_empty() const { return (flag_ == NFM_CONTEXT_DOMAIN_EMPTY); }
    bool is_universe() const { return (get_num_contexts() == 0); }

    const NfmSpace& get_space() const { return space_; }

    size_t get_num_linear() const { return linear_.size(); }
    size_t get_num_non_linear() const { return non_linear_.size(); }
    size_t get_num_contexts() const {
        return (get_num_linear() + get_num_non_linear());
    }

    const std::vector<NfmContext>& get_linear_contexts() const { return linear_; }
    const std::vector<NfmContext>& get_non_linear_contexts() const { return non_linear_; }

    const NfmContext& get_linear_context(size_t idx) const {
        assert(idx < linear_.size());
        return linear_[idx];
    }
    const NfmContext& get_non_linear_context(size_t idx) const {
        assert(idx < non_linear_.size());
        return non_linear_[idx];
    }

    friend bool operator==(const NfmContextDomain& lhs, const NfmContextDomain& rhs);
    friend bool operator!=(const NfmContextDomain& lhs, const NfmContextDomain& rhs);

    /* Hack to make it work with isl */
    void update_space(const NfmSpace& p_space) {
        space_ = p_space;
        for (auto& ctx : linear_) {
            ctx.update_space(p_space);
        }
        for (auto& ctx : non_linear_) {
            ctx.update_space(p_space);
        }
    }

private:
    NfmSpace space_;
    std::vector<NfmContext> linear_;
    std::vector<NfmContext> non_linear_;
    NfmContextDomainFlag flag_;
};

}
}

#endif
