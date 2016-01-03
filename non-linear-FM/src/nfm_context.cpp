#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include "nfm_context.h"

#include "nfm_debug.h"

namespace Nfm {
namespace Internal {

using std::ostream;
using std::ostringstream;
using std::string;

string NfmContext::to_string() const {
    ostringstream stream;
    stream << context_.to_string();
    if (is_eq_) {
        stream << " = 0";
    } else {
        stream << " >= 0";
    }
    return stream.str();
}

ostream& operator<<(ostream& out, const NfmContext& context) {
    return out << context.to_string();
}

bool operator==(const NfmContext& lhs, const NfmContext& rhs) {
    if (lhs.is_eq_ != rhs.is_eq_) {
        return false;
    }
    if (lhs.space_ != rhs.space_) {
        return false;
    }
    if (lhs.context_ != rhs.context_) {
        return false;
    }
    return true;
}

bool operator!=(const NfmContext& lhs, const NfmContext& rhs) {
    return !(lhs == rhs);
}

// Divide all coeffs (including constant) by common term (positive term) if 
// applicable, e.g. (4a)x + (2ab)y + (a^2) z >= 0 becomes (4)x + (2b)y + (a)z >= 0
void NfmContext::simplify() {
    int gcd = context_.content();
    if ((gcd != 0) && (gcd != 1)) {
        if (gcd < 0) {
            gcd = -gcd;
        }
        context_ = context_.exquo(gcd);
    }
}

}
}
