#include "Util.h"
#include "Introspection.h"
#include "Debug.h"
#include "Error.h"
#include <sstream>
#include <map>
#include <cmath>
#include <limits>

namespace Halide {
namespace Internal {

using std::string;
using std::vector;
using std::ostringstream;
using std::map;

string unique_name(char prefix) {
    // arrays with static storage duration should be initialized to zero automatically
    static int instances[256];
    ostringstream str;
    str << prefix << instances[(unsigned char)prefix]++;
    return str.str();
}

bool starts_with(const string &str, const string &prefix) {
    if (str.size() < prefix.size()) return false;
    for (size_t i = 0; i < prefix.size(); i++) {
        if (str[i] != prefix[i]) return false;
    }
    return true;
}

bool ends_with(const string &str, const string &suffix) {
    if (str.size() < suffix.size()) return false;
    size_t off = str.size() - suffix.size();
    for (size_t i = 0; i < suffix.size(); i++) {
        if (str[off+i] != suffix[i]) return false;
    }
    return true;
}

string replace_all(string &str, const string &find, const string &replace) {
    size_t pos = 0;
    while ((pos = str.find(find, pos)) != string::npos) {
        str.replace(pos, find.length(), replace);
        pos += replace.length();
    }
    return str;
}

string unique_name(const string &name, bool user) {
    static map<string, int> known_names;

    // An empty string really does not make sense, but use 'z' as prefix.
    if (name.length() == 0) {
        return unique_name('z');
    }

    // Check the '$' character doesn't appear in the prefix. This lets
    // us separate the name from the number using '$' as a delimiter,
    // which guarantees uniqueness of the generated name, without
    // having to track all names generated so far.
    if (user) {
        for (size_t i = 0; i < name.length(); i++) {
            user_assert(name[i] != '$')
                << "Name \"" << name << "\" is invalid. "
                << "Halide names may not contain the character '$'\n";
        }
    }

    int &count = known_names[name];
    count++;
    if (count == 1) {
        // The very first unique name is the original function name itself.
        return name;
    } else {
        // Use the programmer-specified name but append a number to make it unique.
        ostringstream oss;
        oss << name << '$' << count;
        return oss.str();
    }
}

string base_name(const string &name, char delim) {
    size_t off = name.rfind(delim);
    if (off == string::npos) {
        return name;
    }
    return name.substr(off+1);
}

string make_entity_name(void *stack_ptr, const string &type, char prefix) {
    string name = Introspection::get_variable_name(stack_ptr, type);

    if (name.empty()) {
        return unique_name(prefix);
    } else {
        // Halide names may not contain '.'
        for (size_t i = 0; i < name.size(); i++) {
            if (name[i] == '.') {
                name[i] = ':';
            }
        }
        return unique_name(name);
    }
}

std::vector<std::string> split_string(const std::string &source, const std::string &delim) {
    std::vector<std::string> elements;
    size_t start = 0;
    size_t found = 0;
    while ((found = source.find(delim, start)) != std::string::npos) {
        elements.push_back(source.substr(start, found - start));
        start = found + delim.size();
    }

    // If start is exactly source.size(), the last thing in source is a
    // delimiter, in which case we want to add an empty string to elements.
    if (start <= source.size()) {
        elements.push_back(source.substr(start, std::string::npos));
    }
    return elements;
}

} // Internal

std::pair<int, int> approx_factors_near_sqrt(int n) {
    internal_assert(n >= 0);

    int p = 0, q = 0;
    const int sqrtn = (int)floor(sqrt((float)n));

    // Fast path for square numbers.
    if (sqrtn * sqrtn == n) {
        return std::make_pair(sqrtn, sqrtn);
    }

    for (int f = sqrtn; f > 0; f--) {
        p = n / f;
        q = n / p;
        if (p*q <= n) {
            break;
        }
    }
    internal_assert(p*q <= n);

    if (p > q) std::swap(p, q);
    return std::make_pair(p, q);
}

std::vector<int> approx_factors_near_cubert(int n) {
    internal_assert(n >= 0);

    int p = 0, q = 0, r = 0;
    const int cubertn = (int)floor(pow((float)n, 1.0f/3.0f));

    // Fast path for cube numbers.
    if (cubertn * cubertn * cubertn == n) {
        return {cubertn, cubertn, cubertn};
    }

    for (int f = cubertn; f > 0; f--) {
        const auto qr = approx_factors_near_sqrt(n/f);
        const int qq = qr.first;
        const int rr = qr.second;
        const int pp = n / (qq*rr);
        if (pp > 0 && qq > 0 && rr > 0 && pp*qq*rr <= n) {
            p = pp;
            q = qq;
            r = rr;
            break;
        }
    }
    internal_assert(p*q*r <= n);

    // Sort ascending.
    if (p > r) std::swap(p, r);
    if (q > r) std::swap(q, r);
    if (p > q) std::swap(p, q);
    return {p, q, r};
}


} // Halide
