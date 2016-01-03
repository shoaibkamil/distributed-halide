#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include "nfm_polynom.h"

#include "nfm.h"
#include "nfm_debug.h"
#include "nfm_space.h"

namespace Nfm {
namespace Internal {

using std::map;
using std::ostream;
using std::ostringstream;
using std::string;
using std::vector;

namespace {

inline int div_imp(int a, int b) {
    int bits = sizeof(int)*8;
    int q = a / b;
    int r = a - q * b;
    int bs = b >> (bits - 1);
    int rs = r >> (bits - 1);
    return q - (rs & bs) + (rs & ~bs);
}

// Compute a vector of integer which element is the min of the two. Save the
// result in exp1
void min_exp(vector<int>& exp1, const vector<int>& exp2) {
    assert(exp1.size() == exp2.size());
    for (size_t i = 0; i < exp1.size(); ++i) {
        assert((exp1[i] >= 0) && (exp2[i] >= 0));
        exp1[i] = std::min(exp1[i], exp2[i]);
    }
}

string print_exp(const vector<int>& p_exp, const vector<string>& names) {
    assert(p_exp.size() > 0);
    assert(p_exp.size() == names.size());
    ostringstream stream;
    for (size_t i = 0; i < p_exp.size(); ++i) {
        if (p_exp[i] == 0) {
            continue;
        } else if (p_exp[i] == 1) {
            stream << "*" << names[i];
        } else {
            stream << "*(" << names[i] << "^" << p_exp[i] << ")";
        }
    }
    return stream.str().substr(1);
}

template <typename T>
vector<T> operator+(const vector<T>& a, const vector<T>& b) {
    assert(a.size() == b.size());

    vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}

template <typename T>
vector<T> operator-(const vector<T>& a, const vector<T>& b) {
    assert(a.size() == b.size());

    vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::minus<T>());
    return result;
}

// Compare 2 vectors in reverse direction (from end to begin)
template <typename T>
bool is_less_reverse(const vector<T>& a, const vector<T>& b) {
    assert(a.size() == b.size());
    for (int i = a.size()-1; i >= 0; --i) {
        if (a[i] == b[i]) {
            continue;
        } else if (a[i] < b[i]) {
            return true;
        } else { // a[i] > b[i]
            return false;
        }
    }
    return true;
}

// Return true if the vector 'vec' contains at most one one
bool vector_contains_at_most_one_one(const vector<int> vec) {
    int set = 0;
    for (int val : vec) {
        if (val == 0) {
            continue;
        }
        if ((set > 0) || (val != 1)) {
            return false;
        }
        set += 1;
    }
    return true;
}

}

/******************************** NfmPolyCoeff ********************************/

NfmPolyCoeff::NfmPolyCoeff(const NfmSpace& p_space,
                           const map<vector<int>, int>& p_terms,
                           NfmSign p_sign)
        : space_(p_space)
        , sign_(p_sign) {
    if (p_sign != NFM_ZERO) {
        for (const auto& iter : p_terms) {
            assert(iter.first.size() == p_space.size());
            int coeff = iter.second;
            if (coeff != 0) {
                terms_.emplace(iter.first, coeff);
            }
        }
        if (terms_.empty()) {
            sign_ = NFM_ZERO;
        } else if (is_constant()) {
            // Zero case should have been handled above
            assert(terms_.size() == 1);
            sign_ = nfm_sign_int(terms_.begin()->second);
        }
    }
}

bool NfmPolyCoeff::is_constant() const {
    if (is_zero()) {
        return true;
    }
    if (terms_.size() > 1) {
        return false;
    }
    bool all_zeros = std::all_of(terms_.begin()->first.begin(),
                                 terms_.begin()->first.end(),
                                 [](int i) { return i==0; });
    return all_zeros;
}

bool NfmPolyCoeff::is_one() const {
    return (is_constant() && (terms_.begin()->second == 1));
}

bool NfmPolyCoeff::is_neg_one() const {
    return (is_constant() && (terms_.begin()->second == -1));
}

bool NfmPolyCoeff::is_linear() const {
    for (const auto& iter : terms_) {
        if (iter.second == 0) {
            continue;
        }
        if (!vector_contains_at_most_one_one(iter.first)) {
            return false;
        }
    }
    return true;
}

string NfmPolyCoeff::to_string() const {
    if (is_zero()) {
        return "0";
    }
    ostringstream stream;
    const auto& lastKey = terms_.begin()->first;
    for (auto iter = terms_.rbegin(); iter != terms_.rend(); ++iter) {
        int coeff = iter->second;
        if (coeff == 0) {
            continue;
        }
        bool all_zeros = std::all_of(iter->first.begin(), iter->first.end(),
                                     [](int i) { return i==0; });
        if (all_zeros) { // It's a constant integer
            stream << coeff;
        } else if (coeff == 1) {
            stream << print_exp(iter->first, space_.get_names());
        } else {
            stream << coeff << "*" << print_exp(iter->first, space_.get_names());
        }
        if(iter->first != lastKey) { // If it's not the last element in the map
            stream << " + ";
        }
    }
    return stream.str();
}

string NfmPolyCoeff::to_string_with_sign() const {
    ostringstream stream;
    stream << to_string() << " " << nfm_sign_print_op(sign_);
    return stream.str();
}

ostream& operator<<(ostream& out, const NfmPolyCoeff& poly) {
    return out << poly.to_string();
}

string NfmPolyCoeff::print_sign() const {
    return nfm_sign_print_str(sign_);
}

int NfmPolyCoeff::get_coeff(const vector<int>& p_exp) const {
    assert(p_exp.size() == space_.size());
    const auto& iter = terms_.find(p_exp);
    if (iter != terms_.end()) {
        return iter->second;
    }
    return 0;
}

int NfmPolyCoeff::get_coeff(size_t idx) const {
    assert(idx < space_.size());
    vector<int> p_exp(space_.size(), 0);
    p_exp[idx] = 1;
    return get_coeff(p_exp);
}

std::pair<NfmPolyCoeff, NfmPolyCoeff> NfmPolyCoeff::get_coeff_involving_dim(size_t idx) const {
    assert(idx < space_.size());
    if (is_zero()) {
        return std::make_pair(NfmPolyCoeff(space_), NfmPolyCoeff(space_));
    }

    vector<std::pair<std::pair<NfmPolyCoeff, NfmPolyCoeff>, int>> result;
    for (const auto& iter : terms_) {
        if (iter.first[idx] != 0) {
            vector<int> coeff_exp(iter.first);
            coeff_exp[idx] = 0;
            auto term = NfmPolyCoeff(space_, iter.second, iter.first, NFM_UNKNOWN);
            auto coeff = NfmPolyCoeff(space_, iter.second, coeff_exp, NFM_UNKNOWN);
            result.push_back(std::make_pair(std::make_pair(term, coeff), iter.first[idx]));
        }
    }
    if (result.size() == 0) {
        return std::make_pair(NfmPolyCoeff(space_), NfmPolyCoeff(space_));
    }
    // Sort in ascending order based on the exponent of dim at idx (lower exp first)
    std::sort(result.begin(), result.end(),
        [this] (const std::pair<std::pair<NfmPolyCoeff, NfmPolyCoeff>, int>& pair1,
                const std::pair<std::pair<NfmPolyCoeff, NfmPolyCoeff>, int>& pair2) {
            return pair1.second < pair2.second;
    });
    return result[0].first;
}

NfmSign NfmPolyCoeff::compare(const NfmPolyCoeff& other) const {
    assert(space_ == other.space_);
    NfmPolyCoeff diff = sub(other);
    return diff.get_sign();
}

NfmPolyCoeff NfmPolyCoeff::set_sign(NfmSign sign) const {
    NfmPolyCoeff cst(space_, terms_, sign);
    return cst;
}

NfmPolyCoeff NfmPolyCoeff::add(int constant) const {
    vector<int> const_exp(space_.size(), 0);
    return add(constant, const_exp, nfm_sign_int(constant));
}

NfmPolyCoeff NfmPolyCoeff::add(int coeff, const vector<int>& p_exp, NfmSign p_sign) const {
    assert(p_exp.size() == space_.size());
    if (coeff == 0) {
        return *this;
    }
    map<vector<int>, int> new_terms(terms_);
    new_terms[p_exp] += coeff;
    return NfmPolyCoeff(space_, new_terms, nfm_sign_add(sign_, p_sign));
}

NfmPolyCoeff NfmPolyCoeff::add(const NfmPolyCoeff& other) const {
    assert(space_ == other.space_);
    if (other.is_zero()) {
        return *this;
    }
    map<vector<int>, int> new_terms(terms_);
    for (const auto& iter : other.terms_) {
        new_terms[iter.first] += iter.second;
    }
    return NfmPolyCoeff(space_, new_terms, nfm_sign_add(sign_, other.sign_));
}

NfmPolyCoeff NfmPolyCoeff::neg() const {
    if (is_zero()) {
        return *this;
    }
    map<vector<int>, int> new_terms(terms_);
    for (const auto& iter : terms_) {
        new_terms[iter.first] = -iter.second;
    }
    return NfmPolyCoeff(space_, new_terms, nfm_sign_neg(sign_));
}

NfmPolyCoeff NfmPolyCoeff::sub(int constant) const {
    return add(-constant);
}

NfmPolyCoeff NfmPolyCoeff::sub(int coeff, const vector<int>& p_exp, NfmSign p_sign) const {
    return add(-coeff, p_exp, nfm_sign_neg(p_sign));
}

NfmPolyCoeff NfmPolyCoeff::sub(const NfmPolyCoeff& other) const {
    return add(other.neg());
}

NfmPolyCoeff NfmPolyCoeff::mul(int constant) const {
    vector<int> const_exp(space_.size(), 0);
    return mul(constant, const_exp, nfm_sign_int(constant));
}

NfmPolyCoeff NfmPolyCoeff::mul(int coeff, const vector<int>& p_exp, NfmSign p_sign) const {
    assert(p_exp.size() == space_.size());
    if (is_zero()) {
        return *this;
    }
    if ((coeff == 0) || (p_sign == NFM_ZERO)) {
        return NfmPolyCoeff(space_);
    }
    map<vector<int>, int> new_terms;
    for (const auto& iter : terms_) {
        int new_coeff = iter.second*coeff;
        assert(new_coeff != 0);
        vector<int> new_exp = iter.first + p_exp;
        new_terms.emplace(new_exp, new_coeff);
    }
    return NfmPolyCoeff(space_, new_terms, nfm_sign_mul(sign_, p_sign));
}

NfmPolyCoeff NfmPolyCoeff::mul(const NfmPolyCoeff& other) const {
    assert(space_ == other.space_);
    if (is_zero()) {
        return *this;
    }
    if (other.is_zero()) {
        return NfmPolyCoeff(space_);
    }
    map<vector<int>, int> new_terms;
    for (const auto& iter : terms_) {
        for (const auto& other_iter : other.terms_) {
            int new_coeff = iter.second*other_iter.second;
            assert(new_coeff != 0);
            vector<int> new_exp = iter.first + other_iter.first;
            new_terms.emplace(new_exp, new_coeff);
        }
    }
    return NfmPolyCoeff(space_, new_terms, nfm_sign_mul(sign_, other.sign_));
}

NfmPolyCoeff NfmPolyCoeff::exquo(int constant) const {
    vector<int> const_exp(space_.size(), 0);
    return exquo(constant, const_exp, nfm_sign_int(constant));
}

NfmPolyCoeff NfmPolyCoeff::exquo(int coeff, const vector<int>& p_exp, NfmSign p_sign) const {
    assert((coeff != 0) || (p_sign != NFM_ZERO));
    assert(p_exp.size() == space_.size());
    if (is_zero()) {
        return *this;
    }
    map<vector<int>, int> new_terms;
    for (const auto& iter : terms_) {
        assert(iter.second % coeff == 0); // Should be perfectly divisible
        int new_coeff = iter.second/coeff;
        vector<int> new_exp = iter.first - p_exp;
        // Coefficient should also be perfectly divisible
        assert(std::all_of(new_exp.begin(), new_exp.end(), [](int i) { return i>=0; }));
        new_terms.emplace(new_exp, new_coeff);
    }
    return NfmPolyCoeff(space_, new_terms, nfm_sign_div(sign_, p_sign));
}

// Have to be single term polynomial; otherwise will return erroneous answer
NfmPolyCoeff NfmPolyCoeff::exquo(const NfmPolyCoeff& other) const {
    assert(!other.is_zero());
    assert(other.get_terms().size() == 1);
    return exquo(other.get_terms().begin()->second,
                 other.get_terms().begin()->first,
                 other.get_sign());
}

NfmPolyCoeff NfmPolyCoeff::fdiv(int constant) const {
    assert(constant != 0);
    if (is_zero()) {
        return *this;
    }
    NfmSign new_sign = nfm_sign_div(sign_, nfm_sign_int(constant));
    map<vector<int>, int> new_terms;
    for (const auto& iter : terms_) {
        new_terms.emplace(iter.first, div_imp(iter.second, constant));
    }
    return NfmPolyCoeff(space_, new_terms, new_sign);
}

int NfmPolyCoeff::content() const {
    if (is_zero()) {
        return 0;
    }
    int gcd = 0;
    for (const auto& iter : terms_) {
        gcd = non_neg_gcd(gcd, iter.second);
    }
    return gcd;
}

NfmPolyCoeff NfmPolyCoeff::terms_gcd() const {
    if (is_zero()) {
        return NfmPolyCoeff(space_);
    }
    int const_gcd = 0;
    vector<int> exp_gcd(space_.size(), std::numeric_limits<int>::max());
    for (const auto& iter : terms_) {
        const_gcd = non_neg_gcd(const_gcd, iter.second);
        min_exp(exp_gcd, iter.first);
    }
    return NfmPolyCoeff(const_gcd, exp_gcd, space_.get_names(), NFM_UNKNOWN);
}

NfmPolyCoeff NfmPolyCoeff::drop_term(vector<int> p_exp) const {
    assert(p_exp.size() == space_.size());
    map<vector<int>, int> new_terms(terms_);
    new_terms.erase(p_exp);
    return NfmPolyCoeff(space_, new_terms, sign_);
}

NfmPolyCoeff NfmPolyCoeff::drop_term(size_t idx) const {
    assert(idx < space_.size());
    vector<int> p_exp(space_.size(), 0);
    p_exp[idx] = 1;
    return drop_term(p_exp);
}

bool NfmPolyCoeff::operator<(const NfmPolyCoeff& other) const {
    assert(space_ == other.space_);
    auto iter1 = terms_.rbegin();
    auto iter2 = other.terms_.rbegin();
    for (; iter1 != terms_.rend() && iter2 != other.terms_.rend(); ++iter1, ++iter2) {
        if (iter1->first == iter2->first) {
            if (iter1->second < iter2->second) {
                return true;
            } else if (iter1->second > iter2->second) {
                return false;
            }
        } else if (is_less_reverse(iter1->first, iter2->first)) {
            return true;
        } else {
            return false;
        }
    }
    return true;
}

bool operator==(const NfmPolyCoeff& lhs, const NfmPolyCoeff& rhs) {
    if (lhs.sign_ != rhs.sign_) {
        return false;
    }
    if (lhs.space_ != rhs.space_) {
        return false;
    }
    if (lhs.terms_ != rhs.terms_) {
        return false;
    }
    return true;
}

bool operator!=(const NfmPolyCoeff& lhs, const NfmPolyCoeff& rhs) {
    return !(lhs == rhs);
}


/******************************** NfmPoly ********************************/

NfmPoly::NfmPoly(const NfmSpace& p_coeff_space,
                 const NfmSpace& p_space,
                 const map<vector<int>, NfmPolyCoeff>& p_terms)
        : coeff_space_(p_coeff_space)
        , space_(p_space)
        , zero_coeff_(NfmPolyCoeff(coeff_space_)){
    for (auto iter : p_terms) {
        if (!iter.second.is_zero()) {
            terms_.emplace(iter.first, iter.second);
        }
    }
}

bool NfmPoly::is_zero() const {
    if (terms_.empty()) {
        return true;
    }
    for (const auto& iter : terms_) {
        if (!iter.second.is_zero()) {
            return false;
        }
    }
    return true;
}

bool NfmPoly::is_constant() const {
    if (is_zero()) {
        return true;
    }
    if (terms_.size() > 1) {
        return false;
    }
    bool all_zeros = std::all_of(terms_.begin()->first.begin(),
                                 terms_.begin()->first.end(),
                                 [](int i) { return i==0; });
    return all_zeros;
}

bool NfmPoly::is_linear() const {
    for (const auto& iter : terms_) {
        if (iter.second.is_zero()) {
            continue;
        }
        if (!vector_contains_at_most_one_one(iter.first)) {
            return false;
        }
    }
    return true;
}

string NfmPoly::to_string() const {
    if (is_zero()) {
        return "0";
    }
    ostringstream stream;
    const auto& lastKey = terms_.begin()->first;
    for (auto iter = terms_.rbegin(); iter != terms_.rend(); ++iter) {
        const auto& coeff = iter->second;
        if (coeff.is_zero()) {
            continue;
        }
        bool all_zeros = std::all_of(iter->first.begin(),
                                     iter->first.end(),
                                     [](int i) { return i==0; });
        if (all_zeros) { // It's a constant
            stream << coeff.to_string();
        } else if (coeff.is_one()) {
            stream << print_exp(iter->first, space_.get_names());
        } else if (coeff.is_single_term()) {
            stream << coeff.to_string() << "*"
                   << print_exp(iter->first, space_.get_names());
        } else {
            stream << "(" << coeff.to_string() << ")*"
                   << print_exp(iter->first, space_.get_names());
        }
        if (iter->first != lastKey) { // If it's not the last element in the map
            stream << " + ";
        }
    }
    return stream.str();
}

string NfmPoly::to_string_with_sign() const {
    if (is_zero()) {
        return "0";
    }
    ostringstream stream;
    const auto& lastKey = terms_.rbegin()->first;
    for (const auto& iter : terms_) {
        const auto& coeff = iter.second;
        if (coeff.is_zero()) {
            continue;
        }
        bool all_zeros = std::all_of(iter.first.begin(),
                                     iter.first.end(),
                                     [](int i) { return i==0; });
        if (all_zeros) { // It's a constant
            stream << "(" << coeff.to_string_with_sign() << ")";
        } else if (coeff.is_one()) {
            stream << print_exp(iter.first, space_.get_names());
        } else {
            stream << "(" << coeff.to_string_with_sign() << ")*"
                   << print_exp(iter.first, space_.get_names());
        }
        if (iter.first != lastKey) { // If it's not the last element in the map
            stream << " + ";
        }
    }
    return stream.str();
}

ostream& operator<<(ostream& out, const NfmPoly& poly) {
    return out << poly.to_string();
}

NfmPoly NfmPoly::set_coeff_sign(size_t idx, NfmSign sign) const {
    assert(idx < space_.size());
    map<vector<int>, NfmPolyCoeff> new_terms(terms_);
    vector<int> p_exp(space_.size(), 0);
    p_exp[idx] = 1;
    const auto& it = new_terms.find(p_exp);
    if(it != new_terms.end()) {
        it->second = it->second.set_sign(sign);
    }
    return NfmPoly(coeff_space_, space_, new_terms);
}

const NfmPolyCoeff& NfmPoly::get_coeff(std::vector<int> p_exp) const {
    assert(p_exp.size() == space_.size());
    const auto& it = terms_.find(p_exp);
    if (it != terms_.end()) {
        return it->second;
    }
    return zero_coeff_;
}

const NfmPolyCoeff& NfmPoly::get_coeff(size_t idx) const {
    assert(idx < space_.size());
    vector<int> p_exp(space_.size(), 0);
    p_exp[idx] = 1;
    return get_coeff(p_exp);
}

NfmPolyCoeff& NfmPoly::get_coeff(std::vector<int> p_exp) {
    assert(p_exp.size() == space_.size());
    const auto& it = terms_.find(p_exp);
    if (it != terms_.end()) {
        return it->second;
    }
    return zero_coeff_;
}

NfmPolyCoeff& NfmPoly::get_coeff(size_t idx) {
    assert(idx < space_.size());
    vector<int> p_exp(space_.size(), 0);
    p_exp[idx] = 1;
    return get_coeff(p_exp);
}

NfmPoly NfmPoly::add(int constant) const {
    vector<int> exp_inner(coeff_space_.size(), 0);
    return add(constant, exp_inner, nfm_sign_int(constant));
}

NfmPoly NfmPoly::add(int coeff_inner, const vector<int>& exp_inner, NfmSign p_sign) const {
    vector<int> exp_outer(space_.size(), 0);
    return add(coeff_inner, exp_inner, p_sign, exp_outer);
}

NfmPoly NfmPoly::add(int coeff_inner, const vector<int>& exp_inner,
                     NfmSign p_sign, const vector<int>& exp_outer) const {
    NfmPolyCoeff add_by(coeff_inner, exp_inner, coeff_space_.get_names(), p_sign);
    return add(add_by, exp_outer);
}

NfmPoly NfmPoly::add(const NfmPolyCoeff& constant) const {
    vector<int> exp_outer(space_.size(), 0);
    return add(constant, exp_outer);
}

NfmPoly NfmPoly::add(const NfmPolyCoeff& coeff, const vector<int>& exp_outer) const {
    assert(coeff.get_space() == coeff_space_);
    assert(exp_outer.size() == space_.size());

    if (coeff.is_zero()) {
        return *this;
    }
    map<vector<int>, NfmPolyCoeff> new_terms(terms_);
    auto iter = new_terms.find(exp_outer);
    if (iter != new_terms.end()) {
        iter->second = iter->second.add(coeff);
    } else { // Insert new key
        new_terms.emplace(exp_outer, coeff);
    }
    return NfmPoly(coeff_space_, space_, new_terms);
}

NfmPoly NfmPoly::add(const NfmPoly& other) const {
    assert(space_ == other.space_);
    assert(coeff_space_ == other.coeff_space_);
    if (other.is_zero()) {
        return *this;
    }
    map<vector<int>, NfmPolyCoeff> new_terms(terms_);
    for (const auto& other_iter : other.terms_) {
        auto iter = new_terms.find(other_iter.first);
        if (iter != new_terms.end()) {
            iter->second = iter->second.add(other_iter.second);
        } else { // Insert new key
            new_terms.emplace(other_iter.first, other_iter.second);
        }
    }
    return NfmPoly(coeff_space_, space_, new_terms);
}

NfmPoly NfmPoly::neg() const {
    if (is_zero()) {
        return *this;
    }
    map<vector<int>, NfmPolyCoeff> new_terms;
    for (const auto& iter : terms_) {
        new_terms.emplace(iter.first, iter.second.neg());
    }
    return NfmPoly(coeff_space_, space_, new_terms);
}


NfmPoly NfmPoly::sub(int constant) const {
    return add(-constant);
}

NfmPoly NfmPoly::sub(int coeff_inner, const vector<int>& exp_inner, NfmSign p_sign) const {
    return add(-coeff_inner, exp_inner, nfm_sign_neg(p_sign));
}

NfmPoly NfmPoly::sub(int coeff_inner, const vector<int>& exp_inner,
                     NfmSign p_sign, const vector<int>& exp_outer) const {
    return add(-coeff_inner, exp_inner, nfm_sign_neg(p_sign), exp_outer);
}

NfmPoly NfmPoly::sub(const NfmPolyCoeff& constant) const {
    return add(-constant);
}

NfmPoly NfmPoly::sub(const NfmPolyCoeff& coeff, const vector<int>& exp_outer) const {
    return add(-coeff, exp_outer);
}

NfmPoly NfmPoly::sub(const NfmPoly& other) const {
    return add(other.neg());
}

NfmPoly NfmPoly::mul(int constant) const {
    vector<int> exp_inner(coeff_space_.size(), 0);
    return mul(constant, exp_inner, nfm_sign_int(constant));
}

NfmPoly NfmPoly::mul(int coeff_inner, const vector<int>& exp_inner, NfmSign p_sign) const {
    vector<int> exp_outer(space_.size(), 0);
    return mul(coeff_inner, exp_inner, p_sign, exp_outer);
}

NfmPoly NfmPoly::mul(int coeff_inner, const vector<int>& exp_inner,
                     NfmSign p_sign, const vector<int>& exp_outer) const {
    NfmPolyCoeff mul_by(coeff_inner, exp_inner, coeff_space_.get_names(), p_sign);
    return mul(mul_by, exp_outer);
}

NfmPoly NfmPoly::mul(const NfmPolyCoeff& constant) const {
    vector<int> exp_outer(space_.size(), 0);
    return mul(constant, exp_outer);
}

NfmPoly NfmPoly::mul(const NfmPolyCoeff& coeff, const vector<int>& exp_outer) const {
    assert(coeff.get_space() == coeff_space_);
    assert(exp_outer.size() == space_.size());
    if (is_zero()) {
        return *this;
    }
    if (coeff.is_zero()) {
        return NfmPoly(coeff_space_, space_);
    }
    map<vector<int>, NfmPolyCoeff> new_terms;
    for (const auto& iter : terms_) {
        NfmPolyCoeff new_coeff = iter.second.mul(coeff);
        assert(!new_coeff.is_zero());
        vector<int> new_exp = iter.first + exp_outer;
        new_terms.emplace(new_exp, new_coeff);
    }
    return NfmPoly(coeff_space_, space_, new_terms);
}

NfmPoly NfmPoly::mul(const NfmPoly& other) const {
    assert(space_ == other.space_);
    assert(coeff_space_ == other.coeff_space_);
    if (is_zero()) {
        return *this;
    }
    if (other.is_zero()) {
        return NfmPoly(coeff_space_, space_);
    }
    map<vector<int>, NfmPolyCoeff> new_terms;
    for (const auto& iter : terms_) {
        for (const auto& other_iter : other.terms_) {
            NfmPolyCoeff new_coeff = iter.second.mul(other_iter.second);
            assert(!new_coeff.is_zero());
            vector<int> new_exp = iter.first + other_iter.first;
            new_terms.emplace(new_exp, new_coeff);
        }
    }
    return NfmPoly(coeff_space_, space_, new_terms);
}

NfmPoly NfmPoly::exquo(int constant) const {
    vector<int> exp_inner(coeff_space_.size(), 0);
    return exquo(constant, exp_inner, nfm_sign_int(constant));
}

NfmPoly NfmPoly::exquo(int coeff_inner, const vector<int>& exp_inner, NfmSign p_sign) const {
    vector<int> exp_outer(space_.size(), 0);
    return exquo(coeff_inner, exp_inner, p_sign, exp_outer);
}

NfmPoly NfmPoly::exquo(int coeff_inner, const vector<int>& exp_inner,
                       NfmSign p_sign, const vector<int>& exp_outer) const {
    NfmPolyCoeff div_by(coeff_inner, exp_inner, coeff_space_.get_names(), p_sign);
    return exquo(div_by, exp_outer);
}

NfmPoly NfmPoly::exquo(const NfmPolyCoeff& constant) const {
    vector<int> exp_outer(space_.size(), 0);
    return exquo(constant, exp_outer);
}

NfmPoly NfmPoly::exquo(const NfmPolyCoeff& coeff, const vector<int>& exp_outer) const {
    assert(!coeff.is_zero());
    assert(coeff.get_space() == coeff_space_);
    assert(exp_outer.size() == space_.size());
    if (is_zero()) {
        return *this;
    }
    map<vector<int>, NfmPolyCoeff> new_terms;
    for (const auto& iter : terms_) {
        NfmPolyCoeff new_coeff = iter.second.exquo(coeff);
        vector<int> new_exp = iter.first - exp_outer;
        assert(std::all_of(new_exp.begin(), new_exp.end(), [](int i) { return i>=0; }));
        new_terms.emplace(new_exp, new_coeff);
    }
    return NfmPoly(coeff_space_, space_, new_terms);
}

NfmPoly NfmPoly::fdiv(int constant) const {
    if (is_zero()) {
        return *this;
    }
    map<vector<int>, NfmPolyCoeff> new_terms;
    for (const auto& iter : terms_) {
        NfmPolyCoeff new_coeff = iter.second.fdiv(constant);
        new_terms.emplace(iter.first, new_coeff);
    }
    return NfmPoly(coeff_space_, space_, new_terms);
}

int NfmPoly::content() const {
    if (is_zero()) {
        return 0;
    }
    int gcd = 0;
    for (const auto& iter : terms_) {
        gcd = non_neg_gcd(gcd, iter.second.content());
    }
    return gcd;
}

int NfmPoly::non_constant_content() const {
    if (is_zero()) {
        return 0;
    }
    int gcd = 0;
    bool found_constant = false;
    for (const auto& iter : terms_) {
        bool all_zeros = false;
        if (!found_constant) {
            all_zeros = std::all_of(iter.first.begin(),
                                    iter.first.end(),
                                    [](int i) { return i==0; });
        }
        if (all_zeros) { // It's a constant term
            found_constant = true;
            continue;
        }
        gcd = non_neg_gcd(gcd, iter.second.content());
    }
    return gcd;
}

// p1 and p2 are single term polynomials
NfmPolyCoeff NfmPoly::nfm_poly_coeff_gcd(const NfmPolyCoeff& p1,
        const NfmPolyCoeff& p2) const {
    assert(p1.get_space() == p2.get_space());
    if (p1.is_zero()) {
        return p2;
    }
    if (p2.is_zero()) {
        return p1;
    }

    assert(p1.is_single_term() && p2.is_single_term());
    const auto& p1_terms = p1.get_terms();
    const auto& p2_terms = p2.get_terms();
    assert(p1_terms.size() == 1);
    assert(p2_terms.size() == 1);

    int const_gcd = non_neg_gcd(p1_terms.begin()->second, p2_terms.begin()->second);
    vector<int> exp_gcd(p1_terms.begin()->first);
    min_exp(exp_gcd, p2_terms.begin()->first);
    return NfmPolyCoeff(const_gcd, exp_gcd, p1.get_space().get_names(), NFM_UNKNOWN);
}

NfmPolyCoeff NfmPoly::coeffs_gcd() const {
    if (is_zero()) {
        return NfmPolyCoeff(coeff_space_);
    }
    NfmPolyCoeff coeff_gcd(coeff_space_);
    for (const auto& iter : terms_) {
        coeff_gcd = nfm_poly_coeff_gcd(coeff_gcd, iter.second.terms_gcd());
    }
    return coeff_gcd;
}

NfmPolyCoeff NfmPoly::non_constant_coeffs_gcd() const {
    if (is_zero()) {
        return NfmPolyCoeff(coeff_space_);
    }
    NfmPolyCoeff coeff_gcd(coeff_space_);
    bool found_constant = false;
    for (const auto& iter : terms_) {
        bool all_zeros = false;
        if (!found_constant) {
            all_zeros = std::all_of(iter.first.begin(),
                                    iter.first.end(),
                                    [](int i) { return i==0; });
        }
        if (all_zeros) { // It's a constant term
            found_constant = true;
            continue;
        }
        coeff_gcd = nfm_poly_coeff_gcd(coeff_gcd, iter.second.terms_gcd());
    }
    return coeff_gcd;
}

NfmPoly NfmPoly::drop_term(std::vector<int> p_exp) const {
    assert(p_exp.size() == space_.size());
    map<vector<int>, NfmPolyCoeff> new_terms(terms_);
    new_terms.erase(p_exp);
    return NfmPoly(coeff_space_, space_, new_terms);
}

NfmPoly NfmPoly::drop_term(size_t idx) const {
    assert(idx < space_.size());
    vector<int> p_exp(space_.size(), 0);
    p_exp[idx] = 1;
    return drop_term(p_exp);
}

bool NfmPoly::operator<(const NfmPoly& other) const {
    assert(space_ == other.space_);
    auto iter1 = terms_.rbegin();
    auto iter2 = other.terms_.rbegin();
    for (; iter1 != terms_.rend() && iter2 != other.terms_.rend(); ++iter1, ++iter2) {
        if (iter1->first == iter2->first) {
            if (iter1->second == iter2->second) {
                continue;
            } else if (iter1->second < iter2->second) {
                return true;
            } else { // iter1->second > iter2->second
                return false;
            }
        } else if (is_less_reverse(iter1->first, iter2->first)) {
            return true;
        } else {
            return false;
        }
    }
    return true;
}

bool operator==(const NfmPoly& lhs, const NfmPoly& rhs) {
    if (lhs.coeff_space_ != rhs.coeff_space_) {
        return false;
    }
    if (lhs.space_ != rhs.space_) {
        return false;
    }
    if (lhs.terms_ != rhs.terms_) {
        return false;
    }
    return true;
}

bool operator!=(const NfmPoly& lhs, const NfmPoly& rhs) {
    return !(lhs == rhs);
}

}
}
