#ifndef NFM_POLYNOM_FRAC_H
#define NFM_POLYNOM_FRAC_H

#include <map>
#include <string>
#include <vector>

#include "nfm.h"
#include "nfm_polynom.h"
#include "nfm_space.h"

namespace Nfm {
namespace Internal {

/*
 * Represent a fraction. Denominator can only be a NfmPolyCoeff (denom is
 * always constant)
 */

class NfmPolyFrac {
public:
    NfmPolyFrac(const NfmPoly& num, const NfmPolyCoeff& denom);

    explicit NfmPolyFrac(const NfmPoly& num)
        : NfmPolyFrac(num, NfmPolyCoeff::make_one(num.get_coeff_space())) {}

    NfmPolyFrac(const NfmSpace& p_coeff_space, const NfmSpace& p_space)
        : NfmPolyFrac(NfmPoly::make_zero(p_coeff_space, p_space)) {}

    static NfmPolyFrac make_zero(const NfmSpace& p_coeff_space,
                                 const NfmSpace& p_space) {
        return NfmPolyFrac(p_coeff_space, p_space);
    }

    static NfmPolyFrac make_one(const NfmSpace& p_coeff_space,
                                const NfmSpace& p_space) {
        return NfmPolyFrac(NfmPoly::make_one(p_coeff_space, p_space));
    }

    std::string to_string() const;
    std::string to_string_with_sign() const;
    friend std::ostream& operator<<(std::ostream& out, const NfmPolyFrac& poly);

    const NfmSpace& get_space() const { return space_; }
    const NfmSpace& get_coeff_space() const { return coeff_space_; }

    const NfmPoly& get_num() const { return num_; }
    NfmPoly& get_num() { return num_; }
    const NfmPolyCoeff& get_denom() const { return denom_; }
    NfmPolyCoeff& get_denom() { return denom_; }

    bool is_pos() const;
    bool is_neg() const;
    bool is_zero() const { return num_.is_zero(); }
    bool is_unknown() const;
    bool is_constant() const { return num_.is_constant(); };

    NfmPolyFrac neg() const;
    NfmPolyFrac operator-() const { return neg(); }

    NfmPolyFrac add(const NfmPolyFrac& other) const;
    friend NfmPolyFrac operator+(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs) {
        return lhs.add(rhs);
    }

    NfmPolyFrac sub(const NfmPolyFrac& other) const;
    friend NfmPolyFrac operator-(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs) {
        return lhs.sub(rhs);
    }

    friend bool operator<(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs);
    friend bool operator>(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs);
    friend bool operator<=(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs);
    friend bool operator>=(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs);

    friend bool operator==(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs);
    friend bool operator!=(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs);

private:
    NfmSpace coeff_space_;
    NfmSpace space_;

    NfmPoly num_;
    NfmPolyCoeff denom_;

    enum NfmPolyFracCompare {
        NFM_FRAC_SMALLER_THAN = -1,
        NFM_FRAC_EQUAL = 0,
        NFM_FRAC_BIGGER_THAN = 1,
        NFM_FRAC_UNKNOWN = 2
    };

    // Compare this NfmPolyFrac with 'other'. Return SMALLER_THAN if 'this' is
    // strictly smaller than other, BIGGER_THAN if it's strictly bigger, EQUAL
    // if they are equal. Return UNKNOWN if we can't determine the whether
    // 'this' is equal/smaller than/bigger than 'other'.
    NfmPolyFracCompare compare(const NfmPolyFrac& other) const;
};

}
}

#endif
