#ifndef NFM_CONTEXT_H
#define NFM_CONTEXT_H

#include <assert.h>
#include <string>
#include <utility>

#include "nfm_polynom.h"

namespace Nfm {
namespace Internal {

class NfmContext {
public:
    NfmContext(const NfmPolyCoeff& poly, bool is_eq)
        : is_eq_(is_eq)
        , space_(poly.get_space())
        , context_(poly) {}

    std::string to_string() const;
    std::string to_string_with_sign() const;
    friend std::ostream& operator<<(std::ostream& out, const NfmContext& context);

    const NfmSpace& get_space() const { return space_; }
    const NfmPolyCoeff& get_context() const { return context_; }

    // Return the constant term
    int get_constant() const {
        static std::vector<int> p_exp(space_.size(), 0);
        return context_.get_coeff(p_exp);
    }

    // Divide all coeffs by common term (positive term) if applicable,
    void simplify();

    bool is_equality() const { return is_eq_; }
    bool is_linear() const { return context_.is_linear(); }

    friend bool operator==(const NfmContext& lhs, const NfmContext& rhs);
    friend bool operator!=(const NfmContext& lhs, const NfmContext& rhs);

    // Equality comes first then inequality
    bool operator<( const NfmContext& other) const {
        assert(space_ == other.space_);
        if (is_eq_ && !other.is_eq_) {
            return true;
        } else if (!is_eq_ && other.is_eq_) {
            return false;
        } else { // Both are equalities or both are inequalities
            return context_ < other.context_;
        }
    }

    // Hack to make it work with isl
    void update_space(const NfmSpace& space) {
        space_ = space;
        context_.update_space(space);
    }

private:
    bool is_eq_;
    NfmSpace space_;
    NfmPolyCoeff context_;
};

}
}

#endif
