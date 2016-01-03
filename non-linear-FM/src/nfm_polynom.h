#ifndef NFM_POLYNOM_H
#define NFM_POLYNOM_H

#include <map>
#include <string>
#include <vector>

#include "nfm.h"
#include "nfm_space.h"

namespace Nfm {
namespace Internal {

class NfmSolver;

/*
 * Multivariate polynom used to represent symbolic constant coeff of a multivariate polynom.
 * Example: (2*A+B)*x + 4*y -> in this case (2*A+B) and 4 are represented with NfmPolyCoeff
 * Example: 2*A^2 + 12*A*B + 3 is represented as {[(2, 0), 2], [(1, 1), 12], [(0, 0), 3]}
 */
class NfmPolyCoeff {
public:
    friend class NfmSolver;

    NfmPolyCoeff(const NfmSpace& p_space,
                 const std::map<std::vector<int>, int>& p_terms,
                 NfmSign p_sign);

    // Create zero NfmPolyCoeff
    explicit NfmPolyCoeff(const NfmSpace& p_space)
        : NfmPolyCoeff(p_space, std::map<std::vector<int>, int>(), NFM_ZERO) {}

    NfmPolyCoeff(const std::map<std::vector<int>, int>& p_terms,
                 const std::vector<std::string>& dim_names, NfmSign p_sign)
        : NfmPolyCoeff(NfmSpace(dim_names), p_terms, p_sign) {}

    // Create a one-term polynomial, e.g. if dim_names = {a, b}, 2*a^2 is
    // converted into coeff = 2 and p_exp = {2, 0}
    NfmPolyCoeff(int coeff, const std::vector<int>& p_exp,
                 const std::vector<std::string>& dim_names,
                 NfmSign p_sign)
        : NfmPolyCoeff(NfmSpace(dim_names), {{p_exp, coeff}}, p_sign) {}
    NfmPolyCoeff(const NfmSpace& p_space,
                 int coeff, const std::vector<int>& p_exp,
                 NfmSign p_sign)
        : NfmPolyCoeff(p_space, {{p_exp, coeff}}, p_sign) {}

    std::string to_string() const;
    std::string to_string_with_sign() const;
    std::string print_sign() const;
    friend std::ostream& operator<<(std::ostream& out, const NfmPolyCoeff& poly);

    const NfmSpace& get_space() const { return space_; }
    NfmSign get_sign() const { return sign_; }
    int get_coeff(const std::vector<int>& p_exp) const;
    int get_coeff(size_t idx) const;

    // Return term that involves dim idx (the one with the lowest exp at dim idx
    // is preferrable)
    std::pair<NfmPolyCoeff, NfmPolyCoeff> get_coeff_involving_dim(size_t idx) const;

    const std::map<std::vector<int>, int>& get_terms() const { return terms_; }
    std::map<std::vector<int>, int>& get_terms() { return terms_; }

    // Return the constant term
    int get_constant() const {
        static std::vector<int> p_exp(space_.size(), 0);
        return get_coeff(p_exp);
    }

    // Drop term from poly
    NfmPolyCoeff drop_term(std::vector<int> p_exp) const;
    NfmPolyCoeff drop_term(size_t idx) const;

    bool is_pos() const { return (sign_ == NFM_POSITIVE); }
    bool is_neg() const { return (sign_ == NFM_NEGATIVE); }
    bool is_unknown() const { return (sign_ == NFM_UNKNOWN); }
    bool is_constant() const;
    bool is_zero() const {
        return ((sign_ == NFM_ZERO) || terms_.empty());
    }
    bool is_one() const;
    bool is_neg_one() const;
    // Return true if it has exactly 1 non-zero term
    bool is_single_term() const { return terms_.size() == 1; }
    bool is_univariate() const { return space_.size() <= 1; };
    bool is_multivariate() const { return space_.size() > 1; };

    bool is_linear() const; // Linear polynomial: 3*M -2*N

    NfmSign compare(const NfmPolyCoeff& other) const;

    NfmPolyCoeff set_sign(NfmSign sign) const;

    NfmPolyCoeff add(int constant) const;
    NfmPolyCoeff add(int coeff, const std::vector<int>& p_exp, NfmSign p_sign) const;
    NfmPolyCoeff add(const NfmPolyCoeff& other) const;
    friend NfmPolyCoeff operator+(const NfmPolyCoeff& lhs, const NfmPolyCoeff& rhs) {
        return lhs.add(rhs);
    }
    friend NfmPolyCoeff operator+(const NfmPolyCoeff& poly, int constant) {
        return poly.add(constant);
    }
    friend NfmPolyCoeff operator+(int constant, const NfmPolyCoeff& poly) {
        return poly.add(constant);
    }

    NfmPolyCoeff neg() const;
    NfmPolyCoeff operator-() const { return neg(); }

    NfmPolyCoeff sub(int constant) const;
    NfmPolyCoeff sub(int coeff, const std::vector<int>& p_exp, NfmSign p_sign) const;
    NfmPolyCoeff sub(const NfmPolyCoeff& other) const;
    friend NfmPolyCoeff operator-(const NfmPolyCoeff& lhs, const NfmPolyCoeff& rhs) {
        return lhs.sub(rhs);
    }
    friend NfmPolyCoeff operator-(const NfmPolyCoeff& poly, int constant) {
        return poly.sub(constant);
    }
    friend NfmPolyCoeff operator-(int constant, const NfmPolyCoeff& poly) {
        // Same as -poly + constant
        return poly.neg().add(constant);
    }

    NfmPolyCoeff mul(int constant) const;
    NfmPolyCoeff mul(int coeff, const std::vector<int>& p_exp, NfmSign p_sign) const;
    NfmPolyCoeff mul(const NfmPolyCoeff& other) const;
    friend NfmPolyCoeff operator*(const NfmPolyCoeff& lhs, const NfmPolyCoeff& rhs) {
        return lhs.mul(rhs);
    }
    friend NfmPolyCoeff operator*(const NfmPolyCoeff& poly, int constant) {
        return poly.mul(constant);
    }
    friend NfmPolyCoeff operator*(int constant, const NfmPolyCoeff& poly) {
        return poly.mul(constant);
    }

    // Exact quotient: poly is divisible by the divisor (single term polynomial only)
    NfmPolyCoeff exquo(int constant) const;
    NfmPolyCoeff exquo(int coeff, const std::vector<int>& p_exp, NfmSign p_sign) const;
    NfmPolyCoeff exquo(const NfmPolyCoeff& other) const;

    // Floor div
    NfmPolyCoeff fdiv(int constant) const;

    // Compute integer GCD of the polynomial. Return a positive integer
    int content() const;
    // Compute GCD of the polynomial
    NfmPolyCoeff terms_gcd() const;

    friend bool operator==(const NfmPolyCoeff& lhs, const NfmPolyCoeff& rhs);
    friend bool operator!=(const NfmPolyCoeff& lhs, const NfmPolyCoeff& rhs);

    // For comparison in map/set/etc. Doesn't actually tell you whether
    // one NfmPolyCoeff is less than the other
    bool operator<(const NfmPolyCoeff& other) const;

    // Hack to make it work with isl
    void update_space(const NfmSpace& p_space) { space_ = p_space; }

private:
    NfmSpace space_;
    std::map<std::vector<int>, int> terms_; // Map exponents to its int coeff
    NfmSign sign_;
};


class NfmPoly {
public:
    NfmPoly(const NfmSpace& p_coeff_space,
            const NfmSpace& p_space,
            const std::map<std::vector<int>, NfmPolyCoeff>& p_terms);

    NfmPoly(const NfmSpace& p_coeff_space, const NfmSpace& p_space)
        : NfmPoly(p_coeff_space, p_space, std::map<std::vector<int>, NfmPolyCoeff>()) {}

    NfmPoly(const std::map<std::vector<int>, NfmPolyCoeff>& p_terms,
            const std::vector<std::string>& coeff_names,
            const std::vector<std::string>& names)
        : NfmPoly(NfmSpace(coeff_names), NfmSpace(names), p_terms) {}

    NfmPoly(const std::vector<std::string>& coeff_names,
            const std::vector<std::string>& names)
        : NfmPoly(std::map<std::vector<int>, NfmPolyCoeff>(), coeff_names, names) {}

    std::string to_string() const;
    std::string to_string_with_sign() const;
    friend std::ostream& operator<<(std::ostream& out, const NfmPoly& poly);

    NfmPoly set_coeff_sign(size_t idx, NfmSign sign) const;

    const NfmSpace& get_space() const { return space_; }
    const NfmSpace& get_coeff_space() const { return coeff_space_; }

    // Return the coefficient (a NfmPolyCoeff) of the variable that has p_exp
    // as an exponent
    const NfmPolyCoeff& get_coeff(std::vector<int> p_exp) const;
    NfmPolyCoeff& get_coeff(std::vector<int> p_exp);
    // Return the coefficient of univariate c
    const NfmPolyCoeff& get_coeff(size_t idx) const;
    NfmPolyCoeff& get_coeff(size_t idx);
    // Return the constant term
    const NfmPolyCoeff& get_constant() const {
        static std::vector<int> p_exp(space_.size(), 0);
        return get_coeff(p_exp);
    }
    NfmPolyCoeff& get_constant() {
        static std::vector<int> p_exp(space_.size(), 0);
        return get_coeff(p_exp);
    }

    const std::map<std::vector<int>, NfmPolyCoeff>& get_terms() const { return terms_; }
    std::map<std::vector<int>, NfmPolyCoeff>& get_terms() { return terms_; }

    bool is_zero() const;
    bool is_constant() const;
    bool is_linear() const; // Linear polynomial: (M^2)x + (M-N)y

    NfmPoly add(int constant) const;
    NfmPoly add(int coeff_inner, const std::vector<int>& exp_inner, NfmSign p_sign) const;
    NfmPoly add(int coeff_inner, const std::vector<int>& exp_inner,
                NfmSign p_sign, const std::vector<int>& exp_outer) const;
    NfmPoly add(const NfmPolyCoeff& constant) const; // Add by symbolic const coeff
    NfmPoly add(const NfmPolyCoeff& coeff, const std::vector<int>& exp_outer) const;
    NfmPoly add(const NfmPoly& other) const;
    friend NfmPoly operator+(const NfmPoly& lhs, const NfmPoly& rhs) {
        return lhs.add(rhs);
    }
    friend NfmPoly operator+(const NfmPoly& poly, int constant) {
        return poly.add(constant);
    }
    friend NfmPoly operator+(int constant, const NfmPoly& poly) {
        return poly.add(constant);
    }
    friend NfmPoly operator+(const NfmPoly& poly, const NfmPolyCoeff& constant) {
        return poly.add(constant);
    }
    friend NfmPoly operator+(const NfmPolyCoeff& constant, const NfmPoly& poly) {
        return poly.add(constant);
    }

    NfmPoly neg() const;
    NfmPoly operator-() const { return neg(); }

    NfmPoly sub(int constant) const;
    NfmPoly sub(int coeff_inner, const std::vector<int>& exp_inner, NfmSign p_sign) const;
    NfmPoly sub(int coeff_inner, const std::vector<int>& exp_inner,
                NfmSign p_sign, const std::vector<int>& exp_outer) const;
    NfmPoly sub(const NfmPolyCoeff& constant) const; // Subtract by symbolic const coeff
    NfmPoly sub(const NfmPolyCoeff& coeff, const std::vector<int>& exp_outer) const;
    NfmPoly sub(const NfmPoly& other) const;
    friend NfmPoly operator-(const NfmPoly& lhs, const NfmPoly& rhs) {
        return lhs.sub(rhs);
    }
    friend NfmPoly operator-(const NfmPoly& poly, int constant) {
        return poly.sub(constant);
    }
    friend NfmPoly operator-(int constant, const NfmPoly& poly) {
        // Same as -poly + constant
        return poly.neg().add(constant);
    }
    friend NfmPoly operator-(const NfmPoly& poly, const NfmPolyCoeff& constant) {
        return poly.sub(constant);
    }
    friend NfmPoly operator-(const NfmPolyCoeff& constant, const NfmPoly& poly) {
        return poly.neg().add(constant);
    }

    NfmPoly mul(int constant) const;
    NfmPoly mul(int coeff_inner, const std::vector<int>& exp_inner, NfmSign p_sign) const;
    NfmPoly mul(int coeff_inner, const std::vector<int>& exp_inner,
                NfmSign p_sign, const std::vector<int>& exp_outer) const;
    NfmPoly mul(const NfmPolyCoeff& constant) const; // Multiply by symbolic const coeff
    NfmPoly mul(const NfmPolyCoeff& coeff, const std::vector<int>& exp_outer) const;
    NfmPoly mul(const NfmPoly& other) const;
    friend NfmPoly operator*(const NfmPoly& lhs, const NfmPoly& rhs) {
        return lhs.mul(rhs);
    }
    friend NfmPoly operator*(const NfmPoly& poly, int constant) {
        return poly.mul(constant);
    }
    friend NfmPoly operator*(int constant, const NfmPoly& poly) {
        return poly.mul(constant);
    }
    friend NfmPoly operator*(const NfmPoly& poly, const NfmPolyCoeff& constant) {
        return poly.mul(constant);
    }
    friend NfmPoly operator*(const NfmPolyCoeff& constant, const NfmPoly& poly) {
        return poly.mul(constant);
    }

    // Exact quotient: poly is divisible by the divisor
    NfmPoly exquo(int constant) const;
    NfmPoly exquo(int coeff_inner, const std::vector<int>& exp_inner, NfmSign p_sign) const;
    NfmPoly exquo(int coeff_inner, const std::vector<int>& exp_inner,
                  NfmSign p_sign, const std::vector<int>& exp_outer) const;
    NfmPoly exquo(const NfmPolyCoeff& constant) const; // Exact quotient by symbolic const coeff
    NfmPoly exquo(const NfmPolyCoeff& coeff, const std::vector<int>& exp_outer) const;

    // Floor div
    NfmPoly fdiv(int constant) const;

    // Compute GCD of coefficients of poly. Return a positive integer
    int content() const;
    // Compute integer GCD of the terms' coefficients excluding the constant
    // term. Return a positive integer
    int non_constant_content() const;

    // Compute GCD of the terms' coefficients
    NfmPolyCoeff coeffs_gcd() const;
    // Compute GCD of the terms' coefficients excluding the constant term
    NfmPolyCoeff non_constant_coeffs_gcd() const;

    // Drop term from poly
    NfmPoly drop_term(std::vector<int> p_exp) const;
    NfmPoly drop_term(size_t idx) const;

    friend bool operator==(const NfmPoly& lhs, const NfmPoly& rhs);
    friend bool operator!=(const NfmPoly& lhs, const NfmPoly& rhs);

    // For comparison in map/set/etc. Doesn't actually tell you whether
    // one NfmPoly is less than the other
    bool operator<(const NfmPoly& other) const;

    // Hack to make it work with isl
    void update_coeff_space(const NfmSpace& p_coeff_space) {
        coeff_space_ = p_coeff_space;
        for (auto& iter : terms_) {
            iter.second.update_space(p_coeff_space);
        }
    }
    void update_space(const NfmSpace& p_space) {
        space_ = p_space;
    }

private:
    NfmSpace coeff_space_;
    NfmSpace space_;
    std::map<std::vector<int>, NfmPolyCoeff> terms_; // Map exponents to its NfmPolyCoeff

    NfmPolyCoeff zero_coeff_;

    NfmPolyCoeff nfm_poly_coeff_gcd(const NfmPolyCoeff& p1,
                                    const NfmPolyCoeff& p2) const;
};

}
}

#endif
