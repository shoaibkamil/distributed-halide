#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include "nfm_constraint.h"

#include "nfm_debug.h"

namespace Nfm {
namespace Internal {

using std::ostream;
using std::ostringstream;
using std::string;

string NfmConstraint::to_string() const {
    ostringstream stream;
    stream << constraint_.to_string();
    if (is_eq_) {
        stream << " = 0";
    } else {
        stream << " >= 0";
    }
    return stream.str();
}

string NfmConstraint::to_string_with_sign() const {
    ostringstream stream;
    stream << constraint_.to_string_with_sign();
    if (is_eq_) {
        stream << " = 0";
    } else {
        stream << " >= 0";
    }
    return stream.str();
}

ostream& operator<<(ostream& out, const NfmConstraint& constraint) {
    return out << constraint.to_string();
}

bool operator==(const NfmConstraint& lhs, const NfmConstraint& rhs) {
    if (lhs.is_eq_ != rhs.is_eq_) {
        return false;
    }
    if (lhs.coeff_space_ != rhs.coeff_space_) {
        return false;
    }
    if (lhs.space_ != rhs.space_) {
        return false;
    }
    if (lhs.constraint_ != rhs.constraint_) {
        return false;
    }
    return true;
}

bool operator!=(const NfmConstraint& lhs, const NfmConstraint& rhs) {
    return !(lhs == rhs);
}

NfmConstraint NfmConstraint::set_coeff_sign(size_t idx, NfmSign sign) const {
    NfmPoly new_poly = constraint_.set_coeff_sign(idx, sign);
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq_);
}

NfmConstraint NfmConstraint::add(int constant) const {
    NfmPoly new_poly = constraint_.add(constant);
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq_);
}

NfmConstraint NfmConstraint::add(const NfmPolyCoeff& constant) const {
    NfmPoly new_poly = constraint_.add(constant);
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq_);
}

NfmConstraint NfmConstraint::add(const NfmConstraint& other) const {
    bool is_eq = is_eq_;
    if (!other.is_eq_) {
        is_eq = false;
    }
    NfmPoly new_poly = constraint_.add(other.constraint_);
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq);
}

// Negate all coefficients of the dims. NOTE: this does not
// reverse the inequality
NfmConstraint NfmConstraint::neg() const {
    NfmPoly new_poly = constraint_.neg();
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq_);
}

NfmConstraint NfmConstraint::sub(int constant) const {
    return add(-constant);
}

NfmConstraint NfmConstraint::sub(const NfmPolyCoeff& constant) const {
    return add(constant.neg());
}

// NOTE: this does not reverse the inequality
NfmConstraint NfmConstraint::sub(const NfmConstraint& other) const {
    return add(other.neg());
}

NfmConstraint NfmConstraint::mul(int constant) const {
    NfmPoly new_poly = constraint_.mul(constant);
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq_);
}

NfmConstraint NfmConstraint::mul(const NfmPolyCoeff& coeff) const {
    NfmPoly new_poly = constraint_.mul(coeff);
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq_);
}

NfmConstraint NfmConstraint::exquo(int constant) const {
    NfmPoly new_poly = constraint_.exquo(constant);
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq_);
}

NfmConstraint NfmConstraint::exquo(const NfmPolyCoeff& coeff) const {
    NfmPoly new_poly = constraint_.exquo(coeff);
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq_);
}

NfmConstraint NfmConstraint::fdiv(int constant) const {
    NfmPoly new_poly = constraint_.fdiv(constant);
    return NfmConstraint(coeff_space_, space_, new_poly, is_eq_);
}

int NfmConstraint::non_constant_content() const {
    return constraint_.non_constant_content();
}

// Divide all coeffs (including constant) by common term if applicable,
// e.g. (4a)x + (2ab)y + (a^2) z >= 0 becomes (4)x + (2b)y + (a)z >= 0
NfmConstraint NfmConstraint::simplify() const {
    NfmPolyCoeff gcd = constraint_.coeffs_gcd();
    if (!gcd.is_zero() && !gcd.is_one()) {
        if (gcd.is_unknown()) { // Not safe to simplify
            return *this;
        }
        if (gcd.is_neg()) {
            gcd = -gcd;
        }
        return NfmConstraint(coeff_space_, space_, constraint_.exquo(gcd), is_eq_);
    }
    return *this;
}

// Return True if the Constraint can be determined as infeasible.
bool NfmConstraint::is_infeasible() const {
    if (!is_equality()) {
        if (is_constant()) {
            // Ineq: constant >= 0 where constant < 0 is infeasible
            return get_constant().is_neg();
        }
    } else {
        if (is_constant()) {
            // Eq: constant == 0 where constant != 0 is infeasible
            return !get_constant().is_zero();
        }
    }
    return false;
}

}
}
