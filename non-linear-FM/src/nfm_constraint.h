#ifndef NFM_CONSTRAINT_H
#define NFM_CONSTRAINT_H

#include <string>
#include <utility>

#include "nfm_polynom.h"

namespace Nfm {
namespace Internal {

class NfmSolver;

class NfmConstraint {
public:
    NfmConstraint(const NfmSpace& coeff_space, const NfmSpace& space,
                  const NfmPoly& poly, bool is_eq)
        : is_eq_(is_eq)
        , coeff_space_(coeff_space)
        , space_(space)
        , constraint_(poly)
        , const_exp_(std::vector<int>(space_.size(), 0)) {}

    NfmConstraint(const NfmPoly& poly, bool is_eq)
        : NfmConstraint(poly.get_coeff_space(), poly.get_space(), poly, is_eq) {}

    NfmConstraint to_equality() const {
        return NfmConstraint(coeff_space_, space_, constraint_, true);
    }

    std::string to_string() const;
    std::string to_string_with_sign() const;
    friend std::ostream& operator<<(std::ostream& out, const NfmConstraint& constraint);

    bool is_equality() const { return is_eq_; }
    // It only has (symbolic) constant term
    bool is_constant() const { return constraint_.is_constant(); }

    bool is_linear() const { return constraint_.is_linear(); }
    bool is_infeasible() const;

    NfmConstraint set_coeff_sign(size_t idx, NfmSign sign) const;

    const NfmSpace& get_space() const { return space_; }
    const NfmSpace& get_coeff_space() const { return coeff_space_; }
    const NfmPoly& get_constraint() const { return constraint_; }

    friend bool operator==(const NfmConstraint& lhs, const NfmConstraint& rhs);
    friend bool operator!=(const NfmConstraint& lhs, const NfmConstraint& rhs);

    // Equality comes first then inequality
    bool operator<(const NfmConstraint& other) const {
        assert(coeff_space_ == other.coeff_space_);
        assert(space_ == other.space_);
        if (is_eq_ && !other.is_eq_) {
            return true;
        } else if (!is_eq_ && other.is_eq_) {
            return false;
        } else { // Both are equalities or both are inequalities
            return constraint_ < other.constraint_;
        }
    }

    // Return the constraint's terms
    const std::map<std::vector<int>, NfmPolyCoeff>& get_terms() const {
        return constraint_.get_terms();
    }
    std::map<std::vector<int>, NfmPolyCoeff>& get_terms() {
        return constraint_.get_terms();
    }

    // Return the constant term
    const NfmPolyCoeff& get_constant() const {
        const NfmPolyCoeff& constant = get_coeff(const_exp_);
        return constant;
    }
    NfmPolyCoeff& get_constant() {
        NfmPolyCoeff& constant = get_coeff(const_exp_);
        return constant;
    }

    // Return the coefficient (a NfmPolyCoeff) of the variable that has p_exp
    // as an exponent
    const NfmPolyCoeff& get_coeff(const std::vector<int>& p_exp) const {
        return constraint_.get_coeff(p_exp);
    }
    NfmPolyCoeff& get_coeff(const std::vector<int>& p_exp) {
        return constraint_.get_coeff(p_exp);
    }
    const NfmPolyCoeff& operator[](const std::vector<int>& p_exp) const {
        return get_coeff(p_exp);
    }
    NfmPolyCoeff& operator[](const std::vector<int>& p_exp) {
        return get_coeff(p_exp);
    }

    // Return the coefficient of univariate space_[idx]
    const NfmPolyCoeff& get_coeff(size_t idx) const {
        return constraint_.get_coeff(idx);
    }
    NfmPolyCoeff& get_coeff(size_t idx) {
        return constraint_.get_coeff(idx);
    }
    const NfmPolyCoeff& operator[](size_t idx) const { return get_coeff(idx); }
    NfmPolyCoeff& operator[](size_t idx) { return get_coeff(idx); }

    NfmConstraint add(int constant) const;
    NfmConstraint add(const NfmPolyCoeff& constant) const;
    NfmConstraint add(const NfmConstraint& other) const;
    friend NfmConstraint operator+(const NfmConstraint& lhs, const NfmConstraint& rhs) {
        return lhs.add(rhs);
    }
    friend NfmConstraint operator+(const NfmConstraint& cst, int constant) {
        return cst.add(constant);
    }
    friend NfmConstraint operator+(const NfmConstraint& cst, const NfmPolyCoeff& constant) {
        return cst.add(constant);
    }
    friend NfmConstraint operator+(const NfmPolyCoeff& constant, const NfmConstraint& cst) {
        return cst.add(constant);
    }

    NfmConstraint neg() const;
    NfmConstraint operator-() const { return neg(); }

    NfmConstraint sub(int constant) const;
    NfmConstraint sub(const NfmPolyCoeff& constant) const;
    NfmConstraint sub(const NfmConstraint& other) const;
    friend NfmConstraint operator-(const NfmConstraint& lhs, const NfmConstraint& rhs) {
        return lhs.sub(rhs);
    }
    friend NfmConstraint operator-(const NfmConstraint& cst, int constant) {
        return cst.sub(constant);
    }
    friend NfmConstraint operator-(const NfmConstraint& cst, const NfmPolyCoeff& constant) {
        return cst.sub(constant);
    }
    friend NfmConstraint operator-(const NfmPolyCoeff& constant, const NfmConstraint& cst) {
        // Same as -cst + constant
        return cst.neg().add(constant);
    }

    NfmConstraint mul(int constant) const;
    NfmConstraint mul(const NfmPolyCoeff& other) const;

    NfmConstraint exquo(int constant) const;
    NfmConstraint exquo(const NfmPolyCoeff& other) const;

    NfmConstraint fdiv(int constant) const;

    // Compute integer gcd of all non-zero coefficients of the polynomial
    // constraint (including the constant term)
    int content() const { return constraint_.content(); }
    // Compute integer gcd of all non-zero coefficients of the polynomial
    // constraint (NOT including the constant term)
    int non_constant_content() const;

    // Divide all coeffs by common term if applicable,
    NfmConstraint simplify() const;

    // Hack to make it work with isl
    void update_coeff_space(const NfmSpace& coeff_space) {
        coeff_space_ = coeff_space;
        constraint_.update_coeff_space(coeff_space);
    }
    void update_space(const NfmSpace& space) {
        space_ = space;
        constraint_.update_space(space);
    }

private:
    bool is_eq_;
    NfmSpace coeff_space_;
    NfmSpace space_;
    NfmPoly constraint_;

    std::vector<int> const_exp_;
};

}
}

#endif
