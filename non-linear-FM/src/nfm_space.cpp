#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include "nfm_space.h"

namespace Nfm {
namespace Internal {

using std::ostream;
using std::ostringstream;
using std::string;
using std::vector;

namespace {

vector<string> str_split(const string& s, const string& delim) {
    vector<string> internal;
    auto start = 0U;
    auto end = s.find(delim);
    while (end != string::npos) {
        internal.push_back(s.substr(start, end - start));
        start = end + delim.length();
        end = s.find(delim, start);
    }
    internal.push_back(s.substr(start, end));
    return internal;
}

}

// Create a space where the variables of the space are provided using
// str. str is a comma separated string list, e.g. "a,b,c".
NfmSpace::NfmSpace(const std::string& str) {
    names_ = str_split(str, ",");
}

string NfmSpace::to_string() const {
    ostringstream stream;
    for (size_t i = 0; i < names_.size(); ++i) {
        stream << names_[i];
        if (i != names_.size() - 1) {
            stream << ", ";
        }
    }
    return stream.str();
}

ostream& operator<<(ostream& out, const NfmSpace& space) {
    return out << space.to_string();
}

bool operator==(const NfmSpace& lhs, const NfmSpace& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    return (lhs.names_ == rhs.names_);
}

bool operator!=(const NfmSpace& lhs, const NfmSpace& rhs) {
    return !(lhs == rhs);
}

}
}
