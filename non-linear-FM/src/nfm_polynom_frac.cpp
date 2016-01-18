#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include "nfm_polynom_frac.h"

#include "nfm_debug.h"

namespace Nfm {
namespace Internal {

using std::map;
using std::ostream;
using std::ostringstream;
using std::string;
using std::vector;

NfmPolyFrac::NfmPolyFrac(const NfmPoly& num, const NfmPolyCoeff& denom)
        : coeff_space_(num.get_coeff_space())
        , space_(num.get_space())
        , num_(num)
        , denom_(denom) {
    assert(!denom.is_zero());
    assert(coeff_space_ == space_);

    int gcd = non_neg_gcd(num.content(), denom.content());
    assert(gcd > 0);

    if (gcd != 1) {
        num_ = num_.exquo(gcd);
        denom_ = denom_.exquo(gcd);
    }
}

string NfmPolyFrac::to_string() const {
    if (is_zero()) {
        return "0";
    }
    ostringstream stream;
    if (denom_.is_one()) {
        stream << num_.to_string();
    } else {
        stream << "[" << num_.to_string() << "]/[" << denom_.to_string() << "]";
    }
    return stream.str();
}

string NfmPolyFrac::to_string_with_sign() const {
    if (is_zero()) {
        return "0";
    }
    ostringstream stream;
    if (denom_.is_one()) {
        stream << num_.to_string_with_sign();
    } else {
        stream << "[" << num_.to_string_with_sign() << "]/["
               << denom_.to_string_with_sign() << "]";
    }
    return stream.str();
}

ostream& operator<<(ostream& out, const NfmPolyFrac& frac) {
    return out << frac.to_string();
}

bool NfmPolyFrac::is_pos() const {
    if (is_zero()) {
        return false;
    }
    if (num_.is_pos()) {
        return denom_.is_pos();
    } else if (num_.is_neg()) {
        return denom_.is_neg();
    }
    return false;
}

bool NfmPolyFrac::is_neg() const {
    if (is_zero()) {
        return false;
    }
    if (num_.is_pos()) {
        return denom_.is_neg();
    } else if (num_.is_neg()) {
        return denom_.is_pos();
    }
    return false;
}

bool NfmPolyFrac::is_unknown() const {
    if (is_zero()) {
        return false;
    }
    if (is_pos()) {
        return false;
    }
    if (is_neg()) {
        return false;
    }
    return true;
}

NfmPolyFrac NfmPolyFrac::add(const NfmPolyFrac& other) const {
    assert(space_ == other.space_);
    assert(coeff_space_ == other.coeff_space_);
    NfmPoly new_num = num_*other.denom_ + other.num_*denom_;
    NfmPolyCoeff new_denom = denom_*other.denom_;
    return NfmPolyFrac(new_num, new_denom);
}

NfmPolyFrac NfmPolyFrac::sub(const NfmPolyFrac& other) const {
    return add(-other);
}

NfmPolyFrac NfmPolyFrac::neg() const {
    assert(space_ == other.space_);
    assert(coeff_space_ == other.coeff_space_);
    return NfmPolyFrac(-num_, denom_);
}

NfmPolyFrac::NfmPolyFracCompare NfmPolyFrac::compare(const NfmPolyFrac& other) const {
    assert(space_ == other.space_);
    assert(coeff_space_ == other.coeff_space_);

    NfmPolyFrac diff = sub(other);
    if (diff.is_zero()) {
        return NFM_FRAC_EQUAL;
    } else if (diff.is_pos()) {
        return NFM_FRAC_BIGGER_THAN;
    } else if (diff.is_neg()) {
        return NFM_FRAC_SMALLER_THAN;
    } else {
        return NFM_FRAC_UNKNOWN;
    }
}

bool operator<(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs) {
    NfmPolyFrac::NfmPolyFracCompare comparison = lhs.compare(rhs);
    return (comparison == NfmPolyFrac::NFM_FRAC_SMALLER_THAN);
}

bool operator>(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs) {
    NfmPolyFrac::NfmPolyFracCompare comparison = lhs.compare(rhs);
    return (comparison == NfmPolyFrac::NFM_FRAC_BIGGER_THAN);
}

bool operator<=(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs) {
    NfmPolyFrac::NfmPolyFracCompare comparison = lhs.compare(rhs);
    return (comparison == NfmPolyFrac::NFM_FRAC_SMALLER_THAN) ||
           (comparison == NfmPolyFrac::NFM_FRAC_EQUAL);
}

bool operator>=(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs) {
    NfmPolyFrac::NfmPolyFracCompare comparison = lhs.compare(rhs);
    return (comparison == NfmPolyFrac::NFM_FRAC_BIGGER_THAN) ||
           (comparison == NfmPolyFrac::NFM_FRAC_EQUAL);
}

bool operator==(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs) {
    NfmPolyFrac::NfmPolyFracCompare comparison = lhs.compare(rhs);
    return (comparison == NfmPolyFrac::NFM_FRAC_EQUAL);
}

bool operator!=(const NfmPolyFrac& lhs, const NfmPolyFrac& rhs) {
    return !(lhs == rhs);
}

}
}
