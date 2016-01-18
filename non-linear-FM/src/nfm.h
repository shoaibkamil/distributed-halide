#ifndef NFM_H
#define NFM_H

#include <string>

namespace Nfm {
namespace Internal {

enum NfmSign {
    NFM_NEGATIVE = -1,      // < 0
    NFM_ZERO = 0,           // == 0
    NFM_POSITIVE = 1,       // > 0
    NFM_UNKNOWN = 2,        // ? 0
    NFM_NON_POSITIVE = 3,   // >= 0
    NFM_NON_NEGATIVE = 4    // <= 0
};

inline std::string nfm_sign_print_str(NfmSign sign) {
    if (sign == NFM_ZERO) {
        return "NFM_ZERO";
    } else if (sign == NFM_POSITIVE) {
        return "NFM_POSITIVE";
    } else if (sign == NFM_NEGATIVE) {
        return "NFM_NEGATIVE";
    } else if (sign == NFM_NON_POSITIVE) {
        return "NFM_NON_POSITIVE";
    } else if (sign == NFM_NON_NEGATIVE) {
        return "NFM_NON_NEGATIVE";
    } else {
        return "NFM_UNKNOWN";
    }
}

inline std::string nfm_sign_print_op(NfmSign sign) {
    if (sign == NFM_ZERO) {
        return "= 0";
    } else if (sign == NFM_POSITIVE) {
        return "> 0";
    } else if (sign == NFM_NEGATIVE) {
        return "< 0";
    } else if (sign == NFM_NON_POSITIVE) {
        return ">= 0";
    } else if (sign == NFM_NON_NEGATIVE) {
        return "<= 0";
    } else {
        return "? 0";
    }
}


inline NfmSign nfm_sign_int(int val) {
    NfmSign sign = NFM_UNKNOWN;
    if (val > 0) {
        sign = NFM_POSITIVE;
    } else if (val < 0) {
        sign = NFM_NEGATIVE;
    } else {
        sign = NFM_ZERO;
    }
    return sign;
}

inline NfmSign nfm_sign_neg(NfmSign sign) {
    switch(sign) {
        case NFM_NEGATIVE:
            return NFM_POSITIVE;
        case NFM_ZERO:
            return NFM_ZERO;
        case NFM_POSITIVE:
            return NFM_NEGATIVE;
        case NFM_NON_POSITIVE:
            return NFM_NON_NEGATIVE;
        case NFM_NON_NEGATIVE:
            return NFM_NON_POSITIVE;
        default:
            return NFM_UNKNOWN;
    }
}

inline NfmSign nfm_sign_add(NfmSign sign1, NfmSign sign2) {
    if (sign1 == sign2) {
        return sign1;
    } else if (sign1 == NFM_UNKNOWN || sign2 == NFM_UNKNOWN) {
        return NFM_UNKNOWN;
    } else if ((sign1 == NFM_POSITIVE && sign2 == NFM_ZERO) ||
               (sign1 == NFM_ZERO && sign2 == NFM_POSITIVE)) {
        return NFM_POSITIVE;
    } else if ((sign1 == NFM_NEGATIVE && sign2 == NFM_ZERO) ||
               (sign1 == NFM_ZERO && sign2 == NFM_NEGATIVE)) {
        return NFM_NEGATIVE;
    } else if ((sign1 == NFM_NON_NEGATIVE && sign2 == NFM_ZERO) ||
               (sign1 == NFM_ZERO && sign2 == NFM_NON_NEGATIVE)) {
        return NFM_NON_NEGATIVE;
    } else if ((sign1 == NFM_NON_POSITIVE && sign2 == NFM_ZERO) ||
               (sign1 == NFM_ZERO && sign2 == NFM_NON_POSITIVE)) {
        return NFM_NON_POSITIVE;
    } else if ((sign1 == NFM_NON_NEGATIVE && sign2 == NFM_POSITIVE) ||
               (sign1 == NFM_POSITIVE && sign2 == NFM_NON_NEGATIVE)) {
        return NFM_NON_NEGATIVE;
    } else if ((sign1 == NFM_NON_POSITIVE && sign2 == NFM_NEGATIVE) ||
               (sign1 == NFM_NEGATIVE && sign2 == NFM_NON_POSITIVE)) {
        return NFM_NON_POSITIVE;
    }
    return NFM_UNKNOWN;
}

inline NfmSign nfm_sign_mul(NfmSign sign1, NfmSign sign2) {
    NfmSign new_sign = NFM_UNKNOWN;
    if ((sign1 == NFM_ZERO) || (sign2 == NFM_ZERO)) {
        new_sign = NFM_ZERO;
    } else if ((sign1 == NFM_UNKNOWN) || (sign2 == NFM_UNKNOWN)) {
        new_sign = NFM_UNKNOWN;
    } else if (((sign1 == NFM_POSITIVE) && (sign2 == NFM_POSITIVE)) ||
               ((sign1 == NFM_NEGATIVE) && (sign2 == NFM_NEGATIVE))) {
        new_sign = NFM_POSITIVE;
    } else if (((sign1 == NFM_POSITIVE) && (sign2 == NFM_NEGATIVE)) ||
               ((sign1 == NFM_NEGATIVE) && (sign2 == NFM_POSITIVE))) {
        new_sign = NFM_NEGATIVE;
    } else if (((sign1 == NFM_POSITIVE) && (sign2 == NFM_NON_NEGATIVE)) ||
               ((sign1 == NFM_NON_NEGATIVE) && (sign2 == NFM_POSITIVE)) ||
               ((sign1 == NFM_NON_POSITIVE) && (sign2 == NFM_NEGATIVE)) ||
               ((sign1 == NFM_NEGATIVE) && (sign2 == NFM_NON_POSITIVE))) {
        new_sign = NFM_NON_NEGATIVE;
    } else if (((sign1 == NFM_POSITIVE) && (sign2 == NFM_NON_POSITIVE)) ||
               ((sign1 == NFM_NON_POSITIVE) && (sign2 == NFM_POSITIVE)) ||
               ((sign1 == NFM_NON_NEGATIVE) && (sign2 == NFM_NEGATIVE)) ||
               ((sign1 == NFM_NEGATIVE) && (sign2 == NFM_NON_NEGATIVE))) {
        new_sign = NFM_NON_POSITIVE;
    }
    return new_sign;
}

inline NfmSign nfm_sign_div(NfmSign sign1, NfmSign sign2) {
    return nfm_sign_mul(sign1, sign2);
}

// Compute the GCD of two numbers. Always return a non-neg number
inline int non_neg_gcd(int a, int b) {
    a = abs(a);
    b = abs(b);
    if (a < b) { // Exchange a and b
        a += b;
        b = a - b;
        a -= b;
    }
    if (b == 0) {
      return a;
    }
    while (a % b != 0) {
        a += b;
        b = a - b;
        a -= b;
        b %= a;
    }
    return b;
}

}
}

#endif
