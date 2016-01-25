#include <algorithm>

#include <nfm_solver.h>
#include <nfm_polynom_frac.h>

#include "NfmToHalide.h"

#include "HalideNfmConverter.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "Module.h"
#include "Schedule.h"
#include "Simplify.h"
#include "Substitute.h"

namespace Halide {
namespace Internal {

using namespace Nfm;
using namespace Nfm::Internal;

using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace {

// Iterate over values from last_checked-1 to beginning. If the element is in vector v,
// return that index. Otherwise, return -1
int find_first_index_in_vec(const vector<string> *v, const vector<string>& values,
                            int last_checked) {
    assert(last_checked >= 0 && last_checked <= (int)values.size());
    if ((v != NULL) && (values.size() > 0)) {
        for (int i = last_checked-1; i >= 0; --i) {
            auto iter = std::find(v->begin(), v->end(), values[i]);
            if (iter != v->end()) {
                return i;
            }
        }
    }
    return -1;
}

int find_first_index_in_vec(const vector<string> *v, const vector<Expr>& values,
                            int last_checked) {
    assert(last_checked >= 0 && last_checked <= (int)values.size());
    if ((v != NULL) && (values.size() > 0)) {
        for (int i = last_checked-1; i >= 0; --i) {
            const Variable *var = values[i].as<Variable>();
            assert(var != NULL);
            auto iter = std::find(v->begin(), v->end(), var->name);
            if (iter != v->end()) {
                return i;
            }
        }
    }
    return -1;
}

bool is_val_in_vector(const vector<string> *v, const string& val) {
    if (v != NULL) {
        auto iter = std::find(v->begin(), v->end(), val);
        if (iter != v->end()) {
            return true;
        }
    }
    return false;
}

enum NfmBoundType {
    UNDEFINED = 0,
    EQUAL = 1,
    LOWER_BOUND = 2,
    UPPER_BOUND = 3,
    CONDITION_EQ = 4,
    CONDITION_INEQ = 5
};

enum VarDomainBoundType {
    BOUND_UNDEFINED = 0,
    BOUND_EQUALITY = 1,
    BOUND_INEQUALITY = 2
};

struct NfmBound {
    int dom_idx;
    int idx;
    NfmBoundType type;
    // If lower bound, lsh >= rhs. If upper bound, lhs <= rhs. If equality, lhs==rhs.
    // If condition, it is either lhs==rhs (if CONDITION_EQ) or lhs >= rhs (if CONDITION_INEQ).
    // lhs can't involve any divisions, e.g. x/2 >= y is not allowed, o.t.h, x >= y/2
    // is perfectly valid.
    NfmPoly lhs;
    NfmPolyFrac rhs;

    explicit NfmBound(const NfmSpace& p_coeff_space,
                      const NfmSpace& p_space)
        : dom_idx(-1)
        , idx(-1)
        , type(UNDEFINED)
        , lhs(NfmPoly(p_coeff_space, p_space))
        , rhs(NfmPolyFrac(p_coeff_space, p_space)) {}

    NfmBound(const NfmPoly&& lhs, NfmBoundType type, int idx=-1, int dom_idx=-1)
        : dom_idx(dom_idx)
        , idx(idx)
        , type(type)
        , lhs(std::move(lhs))
        , rhs(NfmPolyFrac::make_zero(lhs.get_coeff_space(), lhs.get_space())) {}

    NfmBound(const NfmPoly&& lhs, const NfmPolyFrac&& rhs, NfmBoundType type, int idx=-1, int dom_idx=-1)
        : dom_idx(dom_idx), idx(idx), type(type), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

    NfmBound(const NfmPoly& lhs, NfmBoundType type, int idx=-1, int dom_idx=-1)
        : dom_idx(dom_idx)
        , idx(idx)
        , type(type)
        , lhs(lhs)
        , rhs(NfmPolyFrac::make_zero(lhs.get_coeff_space(), lhs.get_space())) {}

    NfmBound(const NfmPoly& lhs, const NfmPolyFrac& rhs, NfmBoundType type, int idx=-1, int dom_idx=-1)
        : dom_idx(dom_idx), idx(idx), type(type), lhs(lhs), rhs(rhs) {}

    string to_string() const {
        ostringstream stream;
        switch(type) {
            case EQUAL:
                stream << lhs.to_string() << " == " << rhs.to_string();
                break;
            case LOWER_BOUND:
                stream << lhs.to_string() << " >= " << rhs.to_string();
                break;
            case UPPER_BOUND:
                stream << lhs.to_string() << " <= " << rhs.to_string();
                break;
            case CONDITION_EQ:
                stream << lhs.to_string() << " == " << rhs.to_string();
                break;
            case CONDITION_INEQ:
                stream << lhs.to_string() << " >= " << rhs.to_string();
                break;
            default:
                break;
        }
        return stream.str();
    }

    bool operator==(const NfmBound &other) const {
        if (idx != other.idx) {
            return false;
        }
        if (type != other.type) {
            return false;
        }
        if (lhs != other.lhs) {
            return false;
        }
        if (rhs != other.rhs) {
            return false;
        }
        return true;
    }

    bool operator!=(const NfmBound &other) const {
        if (idx != other.idx) {
            return true;
        }
        if (type != other.type) {
            return true;
        }
        if (lhs != other.lhs) {
            return true;
        }
        if (rhs != other.rhs) {
            return true;
        }
        return false;
    }

    bool is_defined() const { return (type != UNDEFINED); }

    // UNDEFINED is sorted first, followed by EQUAL, LOWER_BOUND, UPPER_BOUND, and
    // CONDITION_EQ, CONDITION_INEQ
    bool operator<(const NfmBound &other) const {
        if (type != other.type) {
            return type < other.type;
        }
        if (idx != other.idx) { // Sort from inner loop to outer loop dimension
            return idx > other.idx;
        }
        if (lhs < other.lhs) {
            return true;
        } else if (lhs == other.lhs) {
            if (rhs < other.rhs) {
                return true;
            } else { // rhs >= other.rhs
                return false;
            }
        } else {
            return false; // lhs > other.lhs
        }
    }
};

typedef vector<NfmBound> AndNfmBounds; // AND of ...

// In Halide, condition bound can be purely inequality/equality (normal conditions)
// or a 'let' assignment
class BoundConditions {
public:
    int dom_idx;
    bool always_false;
    vector<NfmBound> let_conditions; // AND of 'let' conditions
    vector<NfmBound> normal_conditions; // AND of 'normal' conditions

    BoundConditions() : dom_idx(-1), always_false(false) {}

    void sort() {
        std::sort(let_conditions.begin(), let_conditions.end());
        std::sort(normal_conditions.begin(), normal_conditions.end());
    }

    string to_string() const {
        ostringstream stream;
        stream << "let_conditions (" << let_conditions.size() << ")\n";
        for (auto& cond : let_conditions) {
            stream << cond.to_string() << "\n";
        }
        stream << "normal_conditions (" << normal_conditions.size() << ")\n";
        for (auto& cond : normal_conditions) {
            stream << cond.to_string() << "\n";
        }
        return stream.str();
    }

    bool operator<(const BoundConditions &other) const {
        if (always_false && !other.always_false) {
            return true;
        }
        if (!always_false && other.always_false) {
            return false;
        }

        if (let_conditions.size() < other.let_conditions.size()) {
            return true;
        } else if (let_conditions.size() > other.let_conditions.size()) {
            return false;
        }
        for (size_t i = 0; i < let_conditions.size(); ++i) {
            if (let_conditions[i] < other.let_conditions[i]) {
                return true;
            } else if (let_conditions[i] == other.let_conditions[i]) {
                continue;
            } else {
                return false;
            }
        }

        if (normal_conditions.size() < other.normal_conditions.size()) {
            return true;
        } else if (normal_conditions.size() > other.normal_conditions.size()) {
            return false;
        }
        for (size_t i = 0; i < normal_conditions.size(); ++i) {
            if (normal_conditions[i] < other.normal_conditions[i]) {
                return true;
            } else if (normal_conditions[i] == other.normal_conditions[i]) {
                continue;
            } else {
                return false;
            }
        }
        return false;
    }

    bool operator==(const BoundConditions &other) const {
        if (always_false != other.always_false) {
            return false;
        }
        if (let_conditions != other.let_conditions) {
            return false;
        }
        if (normal_conditions != other.normal_conditions) {
            return false;
        }
        return true;
    }

    bool operator!=(BoundConditions &other) const {
        if (always_false != other.always_false) {
            return true;
        }
        if (let_conditions != other.let_conditions) {
            return true;
        }
        if (normal_conditions != other.normal_conditions) {
            return true;
        }
        return false;
    }

    bool is_always_false() const { return always_false; }
    bool is_always_true() const {
        return let_conditions.empty() && normal_conditions.empty();
    }

    bool empty() const {
        return let_conditions.empty() && normal_conditions.empty();
    }

    void insert(const BoundConditions& other) {
        if (other.always_false) {
            always_false = true;
            let_conditions.clear();
            normal_conditions.clear();
        } else {
            for (auto&& bound : other.let_conditions) {
                insert_let_condition(bound);
            }
            for (auto&& bound : other.normal_conditions) {
                insert_normal_condition(bound);
            }
        }
    }

    void insert(const NfmBound& cond, bool is_let) {
        if (is_let) {
            insert_let_condition(cond);
        } else {
            insert_normal_condition(cond);
        }
    }
    void insert(const NfmBound&& cond, bool is_let) {
        if (is_let) {
            insert_let_condition(std::move(cond));
        } else {
            insert_normal_condition(std::move(cond));
        }
    }
    void insert_let_condition(const NfmBound& cond) {
        if (!is_in_let_conditions(cond)) {
            let_conditions.push_back(cond);
        }
    }
    void insert_let_condition(const NfmBound&& cond) {
        if (!is_in_let_conditions(cond)) {
            let_conditions.push_back(std::move(cond));
        }
    }
    void insert_normal_condition(const NfmBound& cond) {
        if (!is_in_normal_conditions(cond)) {
            normal_conditions.push_back(cond);
        }
    }
    void insert_normal_condition(const NfmBound&& cond) {
        if (!is_in_normal_conditions(cond)) {
            normal_conditions.push_back(std::move(cond));
        }
    }
private:
    bool is_in_let_conditions(const NfmBound& cond) const {
        const auto& iter = std::find(let_conditions.begin(), let_conditions.end(), cond);
        return (iter != let_conditions.end());
    }
    bool is_in_normal_conditions(const NfmBound& cond) const {
        const auto& iter = std::find(normal_conditions.begin(), normal_conditions.end(), cond);
        return (iter != normal_conditions.end());
    }
};

// Express lower/upper bound of a dimension within a NfmDomain.
// Also includes the condition for which this bounds are applicable.
// If conditions is empty, it means that it's always true.
// If lb or ub is undefined (empty), it means that it is unbounded, i.e, -infinity <= x
struct VarDomainBound {
    bool feasible;
    string var; // Bound on "var" dimension
    VarDomainBoundType type;
    BoundConditions conditions;
    vector<NfmBound> lower_bounds; // AND of lower bound constraints
    vector<NfmBound> upper_bounds; // AND of upper bound constraints

    explicit VarDomainBound(const string& var) :
        feasible(true), var(var), type(BOUND_UNDEFINED) {}

    string to_string() const {
        ostringstream stream;
        stream << "Conditions: \n" << conditions.to_string() << "\n";
        stream << "Lower bounds: \n";
        for (auto& bound : lower_bounds) {
            stream << "  " << bound.to_string() << "\n";
        }
        stream << "Upper bounds: \n";
        for (auto& bound : upper_bounds) {
            stream << "  " << bound.to_string() << "\n";
        }
        return stream.str();
    }

    bool is_feasible() const { return feasible; }
    // Universe domain (unbounded)
    bool is_universe() const { return (type != BOUND_UNDEFINED); }

    void insert_lower_bound(const NfmBound& bound) {
        if (!is_in_lower_bounds(bound)) {
            lower_bounds.push_back(bound);
        }
    }
    void insert_lower_bound(const NfmBound&& bound) {
        if (!is_in_lower_bounds(bound)) {
            lower_bounds.push_back(std::move(bound));
        }
    }
    void insert_upper_bound(const NfmBound& bound) {
        if (!is_in_upper_bounds(bound)) {
            upper_bounds.push_back(bound);
        }
    }
    void insert_upper_bound(const NfmBound&& bound) {
        if (!is_in_upper_bounds(bound)) {
            upper_bounds.push_back(std::move(bound));
        }
    }
    void insert_condition(const BoundConditions& other) {
        conditions.insert(other);
    }
    void insert_condition(const BoundConditions&& other) {
        conditions.insert(std::move(other));
    }
    void insert_condition(const NfmBound& bound, bool is_let=false) {
        conditions.insert(bound, is_let);
    }
    void insert_condition(const NfmBound&& bound, bool is_let=false) {
        conditions.insert(std::move(bound), is_let);
    }

    bool has_condition() { return !conditions.empty(); }
    bool has_lower_bound() { return !lower_bounds.empty(); }
    bool has_upper_bound() { return !upper_bounds.empty(); }
    bool is_equality() const { return (type == BOUND_EQUALITY); }

private:
    bool is_in_lower_bounds(const NfmBound& cond) const {
        const auto& iter = std::find(lower_bounds.begin(), lower_bounds.end(), cond);
        return (iter != lower_bounds.end());
    }
    bool is_in_upper_bounds(const NfmBound& cond) const {
        const auto& iter = std::find(upper_bounds.begin(), upper_bounds.end(), cond);
        return (iter != upper_bounds.end());
    }
};

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

// Return true if one of the bounds subsumes the other. Update 'rhs' with
// 'lhs' if 'lhs' is the dominant one (by constant term).
// If is_min, then lhs subsumes rhs if lhs < rhs.
// If !is_min, then lhs subsumes rhs if lhs > rhs
bool do_subsume(const NfmBound& lhs, NfmBound& rhs, bool is_min) {
    assert(lhs.idx == rhs.idx);
    assert(lhs.type == rhs.type);
    NfmPolyFrac diff = lhs.rhs - rhs.rhs;
    if (diff.is_unknown()) {
        //debug(0) << "  FALSE do_subsume(min? " << is_min << ") lhs: " << lhs.to_string() << "; rhs: " << rhs.to_string() << "\n";
        return false;
    }
    if (is_min) {
        if (diff.is_neg()) {
            rhs = lhs;
        }
    } else {
        if (diff.is_pos()) {
            rhs = lhs;
        }
    }
    //debug(0) << "  TRUE do_subsume(min? " << is_min << ") lhs: " << lhs.to_string() << "; rhs: " << rhs.to_string() << "\n";
    return true;
}

bool do_subsume(const NfmBound& lhs, AndNfmBounds& bounds, bool is_min) {
    for (auto& rhs : bounds) {
        if (do_subsume(lhs, rhs, is_min)) {
            return true;
        }
    }
    return false;
}

bool do_subsume(const NfmBound& lhs, vector<AndNfmBounds>& bounds, bool is_min) {
    for (auto& rhs : bounds) {
        if (do_subsume(lhs, rhs, is_min)) {
            return true;
        }
    }
    return false;
}


bool do_subsume(const NfmContextDomain& ctx_dom, const NfmBound& lhs,
                NfmBound& rhs, bool is_min) {
    assert(lhs.idx == rhs.idx);
    user_assert(lhs.type == rhs.type) << "lhs: " << lhs.to_string() << "; lhs.type: " << lhs.type
        << "; rhs: " << rhs.to_string() << "; rhs.type: " << rhs.type << ";is_min: " << is_min << "\n";
    NfmPolyFrac diff = lhs.rhs - rhs.rhs;
    //debug(0) << "do_subsume context: " << ctx_dom.to_string() << "\n";
    //debug(0) << "   lhs: " << lhs.to_string() << "; rhs: " << rhs.to_string() << "\n";
    NfmSign sign = NfmSolver::nfm_poly_frac_get_sign(ctx_dom, diff);
    if (is_min) {
        if ((sign == NFM_NEGATIVE) || (sign == NFM_NON_POSITIVE)) {
            // < 0 or <= 0
            rhs = lhs;
        } else if (!((sign == NFM_ZERO) || (sign == NFM_POSITIVE) || (sign == NFM_NON_NEGATIVE))) {
            // NOT(== 0 or >= 0 or > 0)
            //debug(0) << "  FALSE do_subsume(min? " << is_min << ") lhs: " << lhs.to_string() << "; rhs: " << rhs.to_string() << "\n";
            return false;
        }
    } else {
        if ((sign == NFM_POSITIVE) || (sign == NFM_NON_NEGATIVE)) {
            // > 0 or >= 0
            rhs = lhs;
        } else if (!((sign == NFM_ZERO) || (sign == NFM_NEGATIVE) || (sign == NFM_NON_POSITIVE))) {
            // NOT(== 0 or <= 0 or < 0)
            //debug(0) << "  FALSE do_subsume(min? " << is_min << ") lhs: " << lhs.to_string() << "; rhs: " << rhs.to_string() << "\n";
            return false;
        }
    }
    //debug(0) << "  TRUE do_subsume(min? " << is_min << ") lhs: " << lhs.to_string() << "; rhs: " << rhs.to_string() << "\n";
    return true;
}

bool do_subsume(const NfmContextDomain& ctx_dom, const NfmBound& lhs,
                AndNfmBounds& bounds, bool is_min) {
    for (auto& rhs : bounds) {
        if (do_subsume(ctx_dom, lhs, rhs, is_min)) {
            return true;
        }
    }
    return false;
}

bool do_subsume(const NfmContextDomain& ctx_dom, const NfmBound& lhs,
                vector<AndNfmBounds>& bounds, bool is_min) {
    for (auto& rhs : bounds) {
        if (do_subsume(ctx_dom, lhs, rhs, is_min)) {
            return true;
        }
    }
    return false;
}

Expr exponent_to_expr(const vector<int>& p_exp, const vector<Expr>& var) {
    user_assert(p_exp.size() > 0) << "p_exp size should be bigger than 0";
    user_assert(p_exp.size() == var.size()) << "p_exp.size: " << p_exp.size()
        << "; var.size: " << var.size() << "\n";
    Expr expr = 1;
    for (size_t i = 0; i < p_exp.size(); ++i) {
        assert(p_exp[i] >= 0);
        if (p_exp[i] >= 1) {
            for (size_t j = p_exp[i]; j > 0; --j) {
                expr = Mul::make(expr, var[i]);
            }
        }
    }
    expr = simplify(expr);
    return expr;
}

Expr convert_nfm_poly_coeff_to_halide_expr(Type type, const NfmPolyCoeff& poly,
                                           const vector<Expr>& sym_const_vars) {
    //std::cout << "convert_nfm_poly_coeff_to_halide_expr " << poly << "\n";
    Expr result = 0;
    for (auto iter : poly.get_terms()) {
        if (iter.second == 0) {
            continue;
        } else if (iter.second == 1) {
            result = Add::make(result, exponent_to_expr(iter.first, sym_const_vars));
        } else {
            result = Add::make(result,
                Mul::make(iter.second, exponent_to_expr(iter.first, sym_const_vars)));
        }

    }
    //std::cout << "convert_nfm_poly_coeff_to_halide_expr result: " << result << "\n";
    return simplify(result);
}

Expr convert_nfm_poly_to_halide_expr(Type type, const NfmPoly& poly,
                                     const vector<Expr>& sym_const_vars,
                                     const vector<Expr>& dim_vars) {
    //std::cout << "convert_nfm_poly_to_halide_expr " << poly << "\n";
    Expr result = 0;
    for (auto iter : poly.get_terms()) {
        if (iter.second.is_zero()) {
            continue;
        } else if (iter.second.is_one()) {
            result = Add::make(result, exponent_to_expr(iter.first, dim_vars));
        } else {
             Expr coeff_expr = convert_nfm_poly_coeff_to_halide_expr(
                type, iter.second, sym_const_vars);
            result = Add::make(result, Mul::make(coeff_expr, exponent_to_expr(iter.first, dim_vars)));
        }
    }
    //std::cout << "convert_nfm_poly_to_halide_expr result: " << result << "\n";
    return simplify(result);
}

Expr convert_nfm_poly_frac_to_halide_expr(Type type, const NfmPolyFrac& frac,
                                          const vector<Expr>& sym_const_vars,
                                          const vector<Expr>& dim_vars,
                                          NfmBoundType bound_type) {
    //std::cout << "convert_nfm_poly_frac_to_halide_expr " << frac << "\n";
    const NfmPoly& num = frac.get_num();
    const NfmPolyCoeff& denom = frac.get_denom();

    Expr result = convert_nfm_poly_to_halide_expr(type, num, sym_const_vars, dim_vars);
    if (!denom.is_one()) {
        Expr temp = convert_nfm_poly_coeff_to_halide_expr(type, denom, sym_const_vars);
        switch(bound_type) {
            case EQUAL:
                result = cast(type, ceil(result/temp));
                break;
            case LOWER_BOUND:
                result = cast(type, ceil(result/temp));
                break;
            case UPPER_BOUND:
                result = cast(type, floor(result/temp));
                break;
            default:
                assert(false);
                break;
        }
    }
    //std::cout << "convert_nfm_poly_frac_to_halide_expr result: " << result << "\n";
    return simplify(result);
}

Expr convert_bound_to_halide_expr_helper(Type type, const NfmBound& bound,
                                         const vector<Expr>& sym_const_vars,
                                         const vector<Expr>& dim_vars) {
    Expr lhs = convert_nfm_poly_to_halide_expr(type, bound.lhs, sym_const_vars, dim_vars);
    Expr rhs = convert_nfm_poly_frac_to_halide_expr(
        type, bound.rhs, sym_const_vars, dim_vars, bound.type);

    Expr result;
    switch(bound.type) {
        case EQUAL:
            result = (lhs == rhs);
            break;
        case LOWER_BOUND:
            result = (lhs >= rhs);
            break;
        case UPPER_BOUND:
            result = (lhs <= rhs);
            break;
        case CONDITION_EQ:
            result = (lhs == rhs);
            break;
        case CONDITION_INEQ:
            result = (lhs >= rhs);
            break;
        default:
            assert(false);
            break;
    }
    return result;
}

Expr convert_to_let_helper(Type type, vector<NfmBound>& let_conditions, const Expr& val,
        const vector<Expr>& sym_const_vars, const vector<Expr>& dim_vars) {
    assert(let_conditions.size() > 0);

    // Sort from inner loop to outer loop (symbolic constant followed by loop dimension)
    std::sort(let_conditions.begin(), let_conditions.end(),
        [] (const NfmBound& b1, const NfmBound& b2) {
            if (b1.type != b2.type) {
                return b1.type < b2.type;
            }
            return b1.idx > b2.idx;
    });

    Expr expr;
    {
        const NfmBound& bound = let_conditions[0];
        Expr value = simplify(val);
        Expr rhs = convert_nfm_poly_frac_to_halide_expr(type, bound.rhs,
            sym_const_vars, dim_vars, bound.type);
        expr = Let::make(bound.lhs.to_string(), rhs, value);
    }
    for (size_t i = 1; i < let_conditions.size(); ++i) {
        const NfmBound& bound = let_conditions[i];
        Expr rhs = convert_nfm_poly_frac_to_halide_expr(type, bound.rhs,
            sym_const_vars, dim_vars, bound.type);
        expr = Let::make(bound.lhs.to_string(), rhs, expr);
    }
    return expr;
}

Expr convert_to_let_helper(const vector<Expr>& ands, size_t start, const Expr& val) {
    assert(ands.size() > 0);
    assert(ands.size()-1 >= start);

    Expr expr;
    {
        const EQ *cond_eq = ands[ands.size()-1].as<EQ>();
        assert(cond_eq != NULL);
        Expr value = simplify(val);
        const Variable *var = cond_eq->a.as<Variable>();
        expr = Let::make(var->name, cond_eq->b, value);
    }
    for (int i = ands.size()-2; i >= (int)start; --i) {
        const EQ *cond_eq = ands[i].as<EQ>();
        assert(cond_eq != NULL);
        const Variable *var = cond_eq->a.as<Variable>();
        expr = Let::make(var->name, cond_eq->b, expr);
    }
    return expr;
}

NfmBound convert_nfm_context_poly_coeff_to_bound(
        int dom_idx, const NfmSpace& coeff_space, const NfmSpace& space,
        const NfmPolyCoeff& poly, bool is_equality,
        const vector<string> *let_assignments, bool *is_let=NULL) {
    //std::cout << "convert_nfm_poly_coeff_to_halide_expr " << poly << " to halide\n";
    if (is_let != NULL) {
        *is_let = false;
    }
    if (!is_equality) { // Inequality can't be an let assignment; no need to check
        //std::cout << "Converting context linear cst: " << poly << " >= 0\n";
        NfmPoly lhs(coeff_space, space, poly);
        //std::cout << "  lhs: " << lhs << "\n";
        return NfmBound(std::move(lhs), CONDITION_INEQ, -1, dom_idx);
    }

    //std::cout << "Converting context linear cst: " << poly << " == 0\n";

    for (int idx = find_first_index_in_vec(let_assignments, coeff_space.get_names(), coeff_space.size());
            idx >= 0; idx = find_first_index_in_vec(let_assignments, coeff_space.get_names(), idx)) {
        std::pair<NfmPolyCoeff, NfmPolyCoeff> term = poly.get_coeff_involving_dim(idx);
        const auto& coeff = term.second;
        if (!coeff.is_zero()) {
            if (is_let != NULL) {
                *is_let = true;
            }
            NfmPoly lhs(coeff_space, space, term.first.exquo(term.second));

            NfmPolyCoeff rhs_poly = poly-term.first;
            if (coeff.is_one()) {
                NfmPolyFrac rhs(NfmPoly(coeff_space, space, rhs_poly.neg()));
                //std::cout << "  lhs: " << lhs << "; rhs: " << rhs << "\n";
                return NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx+coeff_space.size(), dom_idx);
            } else if (coeff.is_neg_one()) {
                NfmPolyFrac rhs(NfmPoly(coeff_space, space, rhs_poly));
                //std::cout << "  lhs: " << lhs << "; rhs: " << rhs << "\n";
                return NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx+coeff_space.size(), dom_idx);
            } else {
                NfmPolyFrac rhs(NfmPoly(coeff_space, space, rhs_poly.neg()), coeff);
                //std::cout << "  lhs: " << lhs << "; rhs: " << rhs << "\n";
                return NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx+coeff_space.size(), dom_idx);
            }
        }
    }
    NfmPoly lhs(coeff_space, space, poly);
    //std::cout << "  lhs: " << lhs << "\n";
    return NfmBound(std::move(lhs), CONDITION_EQ, -1, dom_idx);
}

BoundConditions convert_nfm_context_domain_to_conditions(
        int dom_idx, const NfmSpace& coeff_space, const NfmSpace& space,
        NfmContextDomain& ctx_dom, const vector<string> *let_assignments) {
    BoundConditions result;
    ctx_dom.simplify();
    if (ctx_dom.is_empty()) {
        result.always_false = true; // Condition is always false
        return result;
    }
    if (ctx_dom.is_universe()) {
        return result;
    }
    const vector<NfmContext>& linear = ctx_dom.get_linear_contexts();
    const vector<NfmContext>& non_linear = ctx_dom.get_non_linear_contexts();

    for (const auto& ctx : linear) {
        bool is_let = false;
        NfmBound bound = convert_nfm_context_poly_coeff_to_bound(
            dom_idx, coeff_space, space, ctx.get_context(), ctx.is_equality(),
            let_assignments, &is_let);
        result.insert(std::move(bound), is_let);
    }
    for (const auto& ctx : non_linear) {
        bool is_let = false;
        NfmBound bound = convert_nfm_context_poly_coeff_to_bound(
            dom_idx, coeff_space, space, ctx.get_context(), ctx.is_equality(),
            let_assignments, &is_let);
        result.insert(std::move(bound), is_let);
    }
    return result;
}

// Only process conditional constraint
NfmBound convert_nfm_constraint_to_bound_condition(
        int dom_idx, const NfmSpace& coeff_space, const NfmSpace& space,
        const NfmContextDomain& ctx_dom, NfmConstraint& cst,
        const vector<string> *let_assignments, bool *is_let) {
    //std::cout << "convert_nfm_constraint_to_bound_condition " << cst << " to halide\n";
    if (is_let != NULL) {
        *is_let = false;
    }
    if (!cst.is_equality()) {
        return NfmBound(cst.get_constraint(), CONDITION_INEQ, -1, dom_idx);
    }

    for (int idx = find_first_index_in_vec(let_assignments, space.get_names(), space.size());
            idx >= 0; idx = find_first_index_in_vec(let_assignments, space.get_names(), idx)) {
        NfmPolyCoeff coeff = cst.get_coeff(idx);
        if (!NfmSolver::nfm_poly_coeff_is_zero(ctx_dom, coeff)) {
            if (is_let != NULL) {
                *is_let = true;
            }
            NfmPoly poly = cst.get_constraint().drop_term(idx);
            NfmPoly lhs(coeff_space, space, NfmPolyCoeff::make_one(coeff_space));
            if (coeff.is_one()) {
                NfmPolyFrac rhs(poly.neg());
                return NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx, dom_idx);
            } else if (coeff.is_neg_one()) {
                NfmPolyFrac rhs(poly);
                return NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx, dom_idx);
            } else {
                NfmPolyFrac rhs(poly.neg(), coeff);
                return NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx, dom_idx);
            }
            break;
        }
    }
    return NfmBound(cst.get_constraint(), CONDITION_EQ, -1, dom_idx);
}

// Ignore constraint not involving the dim (return undefined BoundExpr)
NfmBound convert_nfm_constraint_to_bound_non_condition(
        int dom_idx, const NfmSpace& coeff_space, const NfmSpace& space,
        const NfmContextDomain& ctx_dom, NfmConstraint& cst, int dim_idx) {
    user_assert(dim_idx >= 0 && dim_idx < (int)space.size())
        << "dim_idx: " << dim_idx << "; size: " << space.size() << "\n";
    assert(NfmSolver::nfm_constraint_involves(ctx_dom, cst, dim_idx));

    //std::cout << "convert_nfm_constraint_to_bound_non_condition " << cst << " to halide\n";
    NfmBoundType type = UNDEFINED;
    NfmPolyCoeff coeff = cst.get_coeff(dim_idx);
    NfmPoly poly = cst.get_constraint().drop_term(dim_idx);
    if (NfmSolver::nfm_poly_coeff_is_pos(ctx_dom, coeff)) {
        poly = poly.neg();
    } else {
        assert(NfmSolver::nfm_poly_coeff_is_neg(ctx_dom, coeff)); // Should have had been negative
        coeff = coeff.neg();
    }
    vector<int> p_exp(space.size(), 0);
    p_exp[dim_idx] = 1;
    NfmPoly lhs(coeff_space, space, p_exp, NfmPolyCoeff::make_one(coeff_space));
    NfmPolyFrac rhs(poly);
    if (cst.is_equality()) { // constant*var + ... = 0
        type = EQUAL;
        if (!coeff.is_one()) {
            rhs = NfmPolyFrac(poly, coeff);
        }
    } else if (NfmSolver::nfm_constraint_is_lower_bound(ctx_dom, cst, dim_idx)) { // var + ... >= 0
        type = LOWER_BOUND;
        if (!coeff.is_one()) { // e.g 3*x - 13 >= 0
            assert(NfmSolver::nfm_poly_coeff_is_pos(ctx_dom, coeff));
            rhs = NfmPolyFrac(poly, coeff);
        }
    } else if (NfmSolver::nfm_constraint_is_upper_bound(ctx_dom, cst, dim_idx)) { // -var + ...  >= 0
        type = UPPER_BOUND;
        if (!coeff.is_one()) { // e.g -3*x - 13 >= 0
            assert(NfmSolver::nfm_poly_coeff_is_pos(ctx_dom, coeff));
            rhs = NfmPolyFrac(poly, coeff);
        }
    } else {
        assert(false); // Unknown sign, should not have had reached here
    }
    //std::cout << "  lhs: " << lhs << "; rhs: " << rhs << "\n";
    return NfmBound(std::move(lhs), std::move(rhs), type, dim_idx, dom_idx);
}

void convert_to_value_helper(
        Type type, const NfmUnionDomain& union_dom, const vector<Expr>& sym_const_vars,
        const vector<Expr>& dim_vars, BoundConditions conditions, const AndNfmBounds& bounds,
        bool is_lower_bound, int bound_size, map<Expr, Expr, IRDeepCompare>& val_cond) {

    assert(!conditions.empty());
    assert(bounds.size() > 0);

    Expr value = convert_nfm_poly_frac_to_halide_expr(
            type, bounds[0].rhs, sym_const_vars, dim_vars, bounds[0].type);
    for (size_t i = 1; i < bounds.size(); ++i) {
        Expr temp = convert_nfm_poly_frac_to_halide_expr(
            type, bounds[i].rhs, sym_const_vars, dim_vars, bounds[i].type);
        if (is_lower_bound) { // w >= max(...)
            value = max(value, temp);
        } else { // w <= min(...)
            value = min(value, temp);
        }
    }

    Expr temp1; // cond
    if ((bound_size == 1) && (conditions.normal_conditions.size() == 1) &&
            (conditions.normal_conditions[0].type == CONDITION_EQ)) {
        const auto& bound = conditions.normal_conditions[0];
        assert(bound.dom_idx >= 0 && bound.dom_idx < (int)union_dom.get_num_domains());
        const NfmContextDomain& ctx_dom = union_dom[bound.dom_idx].get_context_domain();
        const auto& coeff_space = bound.lhs.get_coeff_space();
        const auto& space = bound.lhs.get_space();

        int idx;
        for (idx = dim_vars.size()-1; idx >= 0; --idx) {
            if (idx == bound.idx) {
                continue;
            }
            NfmPolyCoeff coeff = bound.lhs.get_coeff(idx);
            if (!NfmSolver::nfm_poly_coeff_is_zero(ctx_dom, coeff)) {
                NfmPoly poly = bound.lhs.drop_term(idx);
                NfmPoly lhs(coeff_space, space, NfmPolyCoeff::make_one(coeff_space));
                if (coeff.is_one()) {
                    NfmPolyFrac rhs(poly.neg());
                    conditions.insert_let_condition(std::move(
                        NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx, bound.dom_idx)));
                } else if (coeff.is_neg_one()) {
                    NfmPolyFrac rhs(poly);
                    conditions.insert_let_condition(std::move(
                        NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx, bound.dom_idx)));
                } else {
                    NfmPolyFrac rhs(poly.neg(), coeff);
                    conditions.insert_let_condition(std::move(
                        NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx, bound.dom_idx)));
                }
                break;
            }
        }
        if (idx == -1) {
            for (idx = sym_const_vars.size()-1; idx >= 0; --idx) {
                const NfmPolyCoeff& poly = bound.lhs.get_constant();
                std::pair<NfmPolyCoeff, NfmPolyCoeff> term = poly.get_coeff_involving_dim(idx);
                const auto& coeff = term.second;
                if (!coeff.is_zero()) {
                    NfmPoly lhs(coeff_space, space, term.first.exquo(term.second));

                    NfmPolyCoeff rhs_poly = poly-term.first;
                    if (coeff.is_one()) {
                        NfmPolyFrac rhs(NfmPoly(coeff_space, space, rhs_poly.neg()));
                        //std::cout << "  lhs: " << lhs << "; rhs: " << rhs << "\n";
                        conditions.insert_let_condition(std::move(
                            NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx+coeff_space.size(), bound.dom_idx)));
                    } else if (coeff.is_neg_one()) {
                        NfmPolyFrac rhs(NfmPoly(coeff_space, space, rhs_poly));
                        //std::cout << "  lhs: " << lhs << "; rhs: " << rhs << "\n";
                        conditions.insert_let_condition(std::move(
                            NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx+coeff_space.size(), bound.dom_idx)));
                    } else {
                        NfmPolyFrac rhs(NfmPoly(coeff_space, space, rhs_poly.neg()), coeff);
                        //std::cout << "  lhs: " << lhs << "; rhs: " << rhs << "\n";
                        conditions.insert_let_condition(std::move(
                            NfmBound(std::move(lhs), std::move(rhs), CONDITION_EQ, idx+coeff_space.size(), bound.dom_idx)));
                    }
                }
            }
        }
    } else {
        for (auto& bound : conditions.normal_conditions) {
            Expr cond = convert_bound_to_halide_expr_helper(type, bound, sym_const_vars, dim_vars);
            if (temp1.defined()) {
                temp1 = And::make(temp1, cond);
            } else {
                temp1 = cond;
            }
        }
    }
    //std::cout << "    temp1: " << temp1 << "\n";
    temp1 = simplify(temp1);
    //std::cout << "    simplify temp1: " << temp1 << "\n";

    Expr temp2; // value
    if (conditions.let_conditions.size() == 0) { // No let
        temp2 = value;
    } else {
        temp2 = convert_to_let_helper(type, conditions.let_conditions, value,
            sym_const_vars, dim_vars);
    }
    //std::cout << "    temp2: " << temp2 << "\n";
    temp2 = simplify(temp2, false);
    //std::cout << "    simplify temp2: " << temp2 << "\n";

    const auto& iter = val_cond.find(temp2);
    if (iter != val_cond.end()) {
        iter->second = Or::make(temp1, iter->second);
    } else {
        val_cond.emplace(temp2, temp1);
    }
}

Expr convert_to_value(Type type, const NfmUnionDomain& union_dom,
                      const vector<Expr>& sym_const_vars, const vector<Expr>& dim_vars,
                      bool is_lower_bound, map<BoundConditions, AndNfmBounds>& bounds) {
    assert(bounds.size() > 0);

    map<Expr, Expr, IRDeepCompare> val_cond; // Map from value to condition
    for (auto& iter : bounds) {
        convert_to_value_helper(type, union_dom, sym_const_vars, dim_vars, iter.first,
                                iter.second, is_lower_bound, bounds.size(), val_cond);
    }
    assert(val_cond.size() > 0);

    Expr expr;
    if (val_cond.size() == 1) {
        if(!val_cond.begin()->first.defined()) { // All conditions only (ineqs)
            assert(val_cond.begin()->second.defined());
            Expr value = simplify(val_cond.begin()->second, false);
            //std::cout << "  value1: " << value << "\n";
            expr = value;
        } else {
            assert(val_cond.begin()->first.defined());
            Expr value = simplify(val_cond.begin()->first, false);
            //std::cout << "  value2: " << value << "\n";
            expr = value;
        }
    } else {
        //std::cout << "\nCONVERT TO LET\n";
        Expr condition, value;
        auto iter = val_cond.rbegin();
        //std::cout << "   cond: " << iter->first << "; value: " << iter->second << "\n";
        // First element
        assert(iter->first.defined());
        assert(iter->second.defined());
        condition = iter->second;
        value = iter->first;
        assert(!expr.defined());
        expr = value;
        ++iter;

        // Second element
        //std::cout << "   cond: " << iter->first << "; value: " << iter->second << "\n";
        assert(iter->first.defined());
        assert(iter->second.defined());
        condition = iter->second;
        value = iter->first;
        assert(expr.defined());
        expr = select(condition, value, expr);
        ++iter;

        for (; iter != val_cond.rend(); ++iter) {
            //std::cout << "   cond: " << iter->first << "; value: " << iter->second << "\n";
            assert(iter->first.defined());
            assert(iter->second.defined());
            condition = iter->second;
            value = iter->first;
            assert(expr.defined());
            expr = select(condition, value, expr);
        }
    }
    return expr;
}

Expr convert_nfm_context_poly_coeff_to_halide_expr(
        Type type, const NfmPolyCoeff& poly, const vector<Expr>& sym_const_vars,
        bool is_equality, const vector<string> *let_assignments, bool *is_let=NULL) {
    //std::cout << "convert_nfm_poly_coeff_to_halide_expr " << poly << " to halide\n";
    if (is_let != NULL) {
        *is_let = false;
    }
    if (!is_equality) { // Inequality can't be an let assignment; no need to check
        //std::cout << "Converting context linear cst: " << poly << " >= 0\n";
        Expr expr = convert_nfm_poly_coeff_to_halide_expr(type, poly, sym_const_vars);
        expr = GE::make(simplify(expr), 0);
        return expr;
    }

    //std::cout << "Converting context linear cst: " << poly << " == 0\n";

    int idx = find_first_index_in_vec(let_assignments, sym_const_vars, sym_const_vars.size());
    if (idx == -1) {
        Expr expr = convert_nfm_poly_coeff_to_halide_expr(type, poly, sym_const_vars);
        expr = EQ::make(simplify(expr), 0);
        return expr;
    }

    Expr lhs;
    Expr rhs;
    for (; idx >= 0; idx = find_first_index_in_vec(let_assignments, sym_const_vars, idx)) {
        std::pair<NfmPolyCoeff, NfmPolyCoeff> term = poly.get_coeff_involving_dim(idx);
        const auto& coeff = term.second;
        if (!coeff.is_zero()) {
            lhs = convert_nfm_poly_coeff_to_halide_expr(type, term.first.exquo(term.second), sym_const_vars);

            NfmPolyCoeff rhs_poly = poly-term.first;
            if (coeff.is_one()) {
                rhs = convert_nfm_poly_coeff_to_halide_expr(type, rhs_poly.neg(), sym_const_vars);
            } else if (coeff.is_neg_one()) {
                rhs = convert_nfm_poly_coeff_to_halide_expr(type, rhs_poly, sym_const_vars);
            } else {
                rhs = convert_nfm_poly_coeff_to_halide_expr(type, rhs_poly.neg(), sym_const_vars);
                Expr denom = convert_nfm_poly_coeff_to_halide_expr(type, coeff, sym_const_vars);
                rhs = cast(type, ceil(rhs/denom));
            }
            if (is_let != NULL) {
                *is_let = true;
            }
            break;
        }
    }
    assert(idx >= 0);
    assert(lhs.defined());
    assert(rhs.defined());
    lhs = simplify(lhs);
    rhs = simplify(rhs);
    Expr expr = EQ::make(lhs, rhs);
    //std::cout << "convert_nfm_context_poly_coeff_to_halide_expr result: " << expr << "\n\n";
    return expr;
}

std::pair<set<Expr, IRDeepCompare>, set<Expr, IRDeepCompare>>
convert_nfm_context_domain_to_halide_expr_pair(Type type, NfmContextDomain& ctx_dom,
                                               const vector<Expr>& sym_const_vars,
                                               const vector<string> *let_assignments) {
    // lets, conditional pairs
    std::pair<set<Expr, IRDeepCompare>, set<Expr, IRDeepCompare>> result;
    ctx_dom.simplify();
    if (ctx_dom.is_empty() || ctx_dom.is_universe()) {
        return result;
    }
    const vector<NfmContext>& linear = ctx_dom.get_linear_contexts();
    const vector<NfmContext>& non_linear = ctx_dom.get_non_linear_contexts();

    for (const auto& ctx : linear) {
        bool is_let = false;
        Expr ctx_expr = convert_nfm_context_poly_coeff_to_halide_expr(
            type, ctx.get_context(), sym_const_vars, ctx.is_equality(),
            let_assignments, &is_let);
        if (is_let) {
            result.first.insert(ctx_expr);
        } else {
            result.second.insert(ctx_expr);
        }
    }
    for (const auto& ctx : non_linear) {
        bool is_let = false;
        Expr ctx_expr = convert_nfm_context_poly_coeff_to_halide_expr(
            type, ctx.get_context(), sym_const_vars, ctx.is_equality(),
            let_assignments, &is_let);
        if (is_let) {
            result.first.insert(ctx_expr);
        } else {
            result.second.insert(ctx_expr);
        }
    }
    return result;
}

Expr convert_nfm_constraint_to_halide_expr(
        Type type, const NfmContextDomain& ctx_dom, const NfmConstraint& cst,
        const vector<Expr>& sym_const_vars, const vector<Expr>& dim_vars,
        const vector<string> *let_assignments, bool *is_let=NULL) {
    //std::cout << "convert_nfm_constraint_to_halide_expr " << cst << "\n";
    if (is_let != NULL) {
        *is_let = false;
    }
    Expr result;
    if (!cst.is_equality()) {
        result = convert_nfm_poly_to_halide_expr(type, cst.get_constraint(),
                                                 sym_const_vars, dim_vars);
        result = GE::make(result, 0);
        return result;
    }

    int idx = find_first_index_in_vec(let_assignments, dim_vars, dim_vars.size());
    if (idx == -1) {
        result = convert_nfm_poly_to_halide_expr(type, cst.get_constraint(),
                                                 sym_const_vars, dim_vars);
        result = EQ::make(result, 0);
        return result;
    }

    for (; idx >= 0; idx = find_first_index_in_vec(let_assignments, dim_vars, idx)) {
        NfmPolyCoeff coeff = cst.get_coeff(idx);
        if (!NfmSolver::nfm_poly_coeff_is_zero(ctx_dom, coeff)) {
            NfmPoly poly = cst.get_constraint().drop_term(idx);
            if (coeff.is_one()) {
                Expr temp = convert_nfm_poly_to_halide_expr(type, poly.neg(), sym_const_vars, dim_vars);
                result = EQ::make(dim_vars[idx], temp);
            } else if (coeff.is_neg_one()) {
                Expr temp = convert_nfm_poly_to_halide_expr(type, poly, sym_const_vars, dim_vars);
                result = EQ::make(dim_vars[idx], temp);
            } else {
                Expr temp = convert_nfm_poly_to_halide_expr(type, poly.neg(), sym_const_vars, dim_vars);
                Expr denom = convert_nfm_poly_coeff_to_halide_expr(
                    type, coeff, sym_const_vars);
                result = EQ::make(dim_vars[idx], cast(type, ceil(simplify(temp/denom))));
            }
            if (is_let != NULL) {
                *is_let = true;
            }
            break;
        }
    }
    assert(idx >= 0);

    //std::cout << "convert_nfm_constraint_to_halide_expr result: " << result << "\n";
    return simplify(result);
}

Expr convert_nfm_domain_to_halide_expr(
        Type type, NfmDomain& dom, const vector<Expr>& sym_const_vars,
        const vector<Expr>& dim_vars,
        const vector<string> *let_assignments) {
    //std::cout << "Converting domain " << dom << " to halide\n";

    NfmContextDomain& ctx_dom = dom.get_context_domain();
    auto ctx_let_cond = convert_nfm_context_domain_to_halide_expr_pair(
        type, ctx_dom, sym_const_vars, let_assignments);

    set<Expr, IRDeepCompare> let_exprs;
    set<Expr, IRDeepCompare> cond_exprs;
    for (auto& expr : ctx_let_cond.first) {
        if (expr.defined() && is_zero(expr)) { // The condition is never true
            return Expr();
        }
        let_exprs.insert(expr);
    }
    for (auto& expr : ctx_let_cond.second) {
        if (expr.defined() && is_zero(expr)) { // The condition is never true
            return Expr();
        }
        cond_exprs.insert(expr);
    }

    const auto& eq_constraints = dom.get_equalities();
    for (size_t i = 0; i < eq_constraints.size(); ++i) {
        bool is_let = false;
        Expr expr = convert_nfm_constraint_to_halide_expr(
            type, ctx_dom, eq_constraints[i], sym_const_vars, dim_vars,
            let_assignments, &is_let);
        if (is_let) {
            let_exprs.insert(expr);
        } else {
            cond_exprs.insert(expr);
        }
    }

    const auto& ineqs_constraints = dom.get_inequalities();
    for (size_t i = 0; i < ineqs_constraints.size(); ++i) {
        bool is_let = false;
        Expr expr = convert_nfm_constraint_to_halide_expr(
            type, ctx_dom, ineqs_constraints[i], sym_const_vars, dim_vars,
            let_assignments, &is_let);
        if (is_let) {
            let_exprs.insert(expr);
        } else {
            cond_exprs.insert(expr);
        }
    }

    Expr and_expr;
    for (auto& expr : cond_exprs) {
        if (and_expr.defined()) {
            and_expr = And::make(and_expr, expr);
        } else {
            and_expr = expr;
        }
    }
    if (let_exprs.size() > 0) {
        and_expr = convert_to_let_helper(
            vector<Expr>(let_exprs.begin(), let_exprs.end()), 0, and_expr);
    }
    and_expr = simplify(and_expr);
    return and_expr;
}

VarDomainBound convert_nfm_domain_to_bound(
        int dom_idx, NfmDomain& dom, const string& dim_name, int dim_idx,
        const vector<string> *let_assignments) {
    assert(dim_name.length() >= 0);
    assert(dim_idx >= 0);

    //std::cout << "\nConverting domain " << dom << " to halide\n";
    VarDomainBound result(dim_name);
    if (dom.is_empty()) {
        result.feasible = false;
        return result;
    }
    NfmContextDomain& ctx_dom = dom.get_context_domain();
    BoundConditions ctx_conds = convert_nfm_context_domain_to_conditions(
        dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, let_assignments);
    if (ctx_conds.is_always_false()) { // The condition is never true
        result.feasible = false;
        return result;
    }
    result.insert_condition(std::move(ctx_conds));
    if (dom.is_universe()) {
        return result;
    }

    // Get the lower and upper bounds
    for (auto& eq : dom.get_equalities() ) {
        if (!NfmSolver::nfm_constraint_involves(ctx_dom, eq, dim_idx)) {
            bool is_let = false;
            NfmBound bound = convert_nfm_constraint_to_bound_condition(
                dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, eq,
                let_assignments, &is_let);
            assert(bound.type == CONDITION_EQ);
            //std::cout << "convert_nfm_domain_to_bound EQ cond: " << bound.to_string() << "\n";
            result.insert_condition(std::move(bound), is_let);
        } else {
            NfmBound bound = convert_nfm_constraint_to_bound_non_condition(
                dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, eq, dim_idx);
            assert(bound.type == EQUAL);
            user_assert(!result.has_lower_bound() && !result.has_upper_bound())
                << "can only have one equality\n";
            // Can have different eq values given different conditions, but
            // within a condition, there is only 1 possible lb/ub value
            //std::cout << "convert_nfm_domain_to_bound EQ non_cond: " << bound.to_string() << "\n";
            result.insert_lower_bound(bound);
            result.insert_upper_bound(bound);
            result.type = BOUND_EQUALITY;
        }
    }

    for (auto& ineq : dom.get_inequalities()) {
        if (!NfmSolver::nfm_constraint_involves(ctx_dom, ineq, dim_idx)) {
            bool is_let = false;
            NfmBound bound = convert_nfm_constraint_to_bound_condition(
                dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, ineq,
                let_assignments, &is_let);
            assert(bound.type == CONDITION_INEQ);
            assert(is_let == false);
            //std::cout << "convert_nfm_domain_to_bound INEQ cond: " << bound.to_string() << "\n";
            result.insert_condition(std::move(bound), is_let);
        } else {
            NfmBound bound = convert_nfm_constraint_to_bound_non_condition(
                dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, ineq, dim_idx);
            switch (bound.type) {
                case LOWER_BOUND:
                    if (result.type == BOUND_EQUALITY) {
                        // Since we found an equality, it can only have one answer.
                        continue;
                    }
                    //std::cout << "convert_nfm_domain_to_bound INEQ non_cond: " << bound.to_string() << "\n";
                    result.insert_lower_bound(std::move(bound));
                    result.type = BOUND_INEQUALITY;
                    break;
                case UPPER_BOUND:
                    if (result.type == BOUND_EQUALITY) {
                        // Since we found an equality, it can only have one answer.
                        continue;
                    }
                    //std::cout << "convert_nfm_domain_to_bound INEQ non_cond: " << bound.to_string() << "\n";
                    result.insert_upper_bound(std::move(bound));
                    result.type = BOUND_INEQUALITY;
                    break;
                default:
                    assert(false); // Shouldn't have had reached here
                    break;
            }
        }
    }
    return result;
}

vector<VarDomainBound> convert_nfm_domain_to_bound_box_helper(
        int dom_idx, NfmDomain& dom, const std::vector<std::string>& box_dims,
        int start_dim_idx, int end_dim_idx, const vector<string> *let_assignments) {
    assert(box_dims.size() >= 0);
    assert(start_dim_idx <= end_dim_idx);
    assert(start_dim_idx >= 0 && end_dim_idx >= 0);

    //std::cout << "\nConverting domain " << dom << " to halide\n";

    vector<VarDomainBound> results;
    for (size_t i = 0; i < box_dims.size(); ++i) {
        results.push_back(VarDomainBound(box_dims[i]));
    }

    if (dom.is_empty()) {
        for (size_t i = 0; i < box_dims.size(); ++i) {
            results[i].feasible = false;
        }
        return results;
    }

    NfmContextDomain& ctx_dom = dom.get_context_domain();
    BoundConditions ctx_conds = convert_nfm_context_domain_to_conditions(
        dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, let_assignments);
    if (ctx_conds.is_always_false()) { // The condition is never true
        for (size_t i = 0; i < box_dims.size(); ++i) {
            results[i].feasible = false;
        }
        return results;
    }
    BoundConditions condition = std::move(ctx_conds);

    if (dom.is_universe()) {
        for (size_t i = 0; i < results.size(); ++i) {
            results[i].insert_condition(condition);
        }
        return results;
    }

    vector<bool> eq_is_visited(dom.get_num_equalities(), false);
    vector<bool> ineq_is_visited(dom.get_num_inequalities(), false);
    for (int dim_idx = end_dim_idx; dim_idx >= start_dim_idx; --dim_idx) {
        VarDomainBound& result = results[dim_idx];
        // Get the lower and upper bounds
        for (size_t j = 0; j < dom.get_num_equalities(); ++j) {
            auto& eq = dom.get_equality(j);
            if (eq_is_visited[j] || !NfmSolver::nfm_constraint_involves(ctx_dom, eq, dim_idx)) {
                continue;
            }
            eq_is_visited[j] = true;

            NfmBound bound = convert_nfm_constraint_to_bound_non_condition(
                dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, eq, dim_idx);
            assert(bound.type == EQUAL);
            user_assert(!result.has_lower_bound() && !result.has_upper_bound())
                << "can only have one equality\n";
            // Can have different eq values given different conditions, but
            // within a condition, there is only 1 possible lb/ub value
            //std::cout << "convert_nfm_domain_to_bound EQ non_cond: " << bound.to_string() << "\n";
            result.insert_lower_bound(bound);
            result.insert_upper_bound(bound);
            result.type = BOUND_EQUALITY;
        }

        for (size_t j = 0; j < dom.get_num_inequalities(); ++j) {
            auto& ineq = dom.get_inequality(j);
            if (ineq_is_visited[j] || !NfmSolver::nfm_constraint_involves(ctx_dom, ineq, dim_idx)) {
                continue;
            }
            ineq_is_visited[j] = true;

            NfmBound bound = convert_nfm_constraint_to_bound_non_condition(
                dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, ineq, dim_idx);
            switch (bound.type) {
                case LOWER_BOUND:
                    if (result.type == BOUND_EQUALITY) {
                        // Since we found an equality, it can only have one answer.
                        continue;
                    }
                    //std::cout << "convert_nfm_domain_to_bound INEQ non_cond: " << bound.to_string() << "\n";
                    result.insert_lower_bound(std::move(bound));
                    result.type = BOUND_INEQUALITY;
                    break;
                case UPPER_BOUND:
                    if (result.type == BOUND_EQUALITY) {
                        // Since we found an equality, it can only have one answer.
                        continue;
                    }
                    //std::cout << "convert_nfm_domain_to_bound INEQ non_cond: " << bound.to_string() << "\n";
                    result.insert_upper_bound(std::move(bound));
                    result.type = BOUND_INEQUALITY;
                    break;
                default:
                    assert(false); // Shouldn't have had reached here
                    break;
            }
        }
    }

    for (size_t i = 0; i < eq_is_visited.size(); ++i) {
        if (!eq_is_visited[i]) {
            auto& eq = dom.get_equality(i);

            bool is_let = false;
            NfmBound bound = convert_nfm_constraint_to_bound_condition(
                dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, eq,
                let_assignments, &is_let);
            assert(bound.type == CONDITION_EQ);
            //std::cout << "convert_nfm_domain_to_bound EQ cond: " << bound.to_string() << "\n";
            condition.insert(std::move(bound), is_let);
        }
    }
    for (size_t i = 0; i < ineq_is_visited.size(); ++i) {
        if (!ineq_is_visited[i]) {
            auto& ineq = dom.get_inequality(i);

            bool is_let = false;
            NfmBound bound = convert_nfm_constraint_to_bound_condition(
                dom_idx, dom.get_coeff_space(), dom.get_space(), ctx_dom, ineq,
                let_assignments, &is_let);
            assert(bound.type == CONDITION_INEQ);
            assert(is_let == false);
            //std::cout << "convert_nfm_domain_to_bound INEQ cond: " << bound.to_string() << "\n";
            condition.insert(std::move(bound), is_let);
        }
    }

    for (size_t i = 0; i < results.size(); ++i) {
        results[i].insert_condition(condition);
    }
    return results;
}

}

Expr convert_nfm_union_domain_to_halide_expr(Type type, NfmUnionDomain& union_dom,
                                             const vector<string> *let_assignments,
                                             const map<string, Expr> *expr_substitutions,
                                             const vector<pair<string, Expr>> *let_substitutions) {
    if (union_dom.is_empty()) {
        return 1 < 0; // Empty union dom: infeasible constraint
    }
    if (union_dom.is_universe()) {
        return 1 > 0;
    }

    const auto& coeff_space = union_dom.get_coeff_space();
    const auto& space = union_dom.get_space();
    vector<Expr> sym_const_vars;
    for (size_t i = 0; i < coeff_space.size(); ++i) {
        Expr var = Variable::make(type, coeff_space.get_name(i));
        sym_const_vars.push_back(std::move(var));
    }

    vector<Expr> dim_vars;
    for (size_t i = 0; i < space.size(); ++i) {
        Expr var = Variable::make(type, space.get_name(i));
        dim_vars.push_back(std::move(var));
    }

    auto& domains = union_dom.get_domains();
    Expr or_expr;
    for (size_t i = 0; i < domains.size(); ++i) {
        Expr expr = convert_nfm_domain_to_halide_expr(type, domains[i],
            sym_const_vars, dim_vars, let_assignments);
        assert(expr.defined());
        if (or_expr.defined()) {
            or_expr = Or::make(or_expr, expr);
        } else {
            or_expr = expr;
        }
    }
    if (!or_expr.defined()) {
        return or_expr;
    }
    if ((let_substitutions != NULL) && !let_substitutions->empty()) {
        or_expr = Let::make((*let_substitutions)[let_substitutions->size()-1].first,
            (*let_substitutions)[let_substitutions->size()-1].second, or_expr);
        for (int i = let_substitutions->size()-2; i >= 0; --i) {
            or_expr = Let::make((*let_substitutions)[i].first, (*let_substitutions)[i].second, or_expr);
        }
    }
    if (expr_substitutions != NULL) {
        or_expr = simplify(substitute(*expr_substitutions, or_expr));
    } else {
        or_expr = simplify(or_expr);
    }
    return or_expr;
}

// Convert the union_domain into interval of dim_name, i.e. lb(...) <= dim_name <= ub(...)
Interval convert_nfm_union_domain_to_halide_interval(
        Type type, const Nfm::Internal::NfmUnionDomain& p_union_dom,
        const string& dim_name, const vector<string> *let_assignments,
        const map<string, Expr> *expr_substitutions,
        const vector<pair<string, Expr>> *let_substitutions) {
    // TODO: Not really clear what should be returned if the union dom is empty.
    // Technically, we could return something like -4 >= x >= 0, but it seems
    // that it's a valid interval in Halide (???)
    if (p_union_dom.is_empty() || p_union_dom.is_universe()) {
        return Interval(dim_name);
    }
    const NfmSpace& coeff_space = p_union_dom.get_coeff_space();
    const NfmSpace& space = p_union_dom.get_space();
    int dim_idx = space.get_index(dim_name);
    if (dim_idx == -1) { // The dimension doesn't exist
        return Interval(dim_name);
    }

    vector<Expr> sym_const_vars;
    for (size_t i = 0; i < coeff_space.size(); ++i) {
        Expr var = Variable::make(type, coeff_space.get_name(i));
        sym_const_vars.push_back(std::move(var));
    }

    vector<Expr> dim_vars;
    for (size_t i = 0; i < space.size(); ++i) {
        Expr var = Variable::make(type, space.get_name(i));
        dim_vars.push_back(std::move(var));
    }

    // Project out everything after this dim so that it doesn't depend on
    // inner loop vars
    NfmUnionDomain union_dom =
        NfmSolver::nfm_union_domain_eliminate_dims(p_union_dom, (size_t)dim_idx+1);
    //std::cout << "Union domain after projection: \n" << union_dom << "\n";

    union_dom = NfmSolver::nfm_union_domain_classify_unknown_coefficient(union_dom, (size_t)dim_idx);

    Interval result(dim_name);

    // Lower and upper bounds without a condition (usually appers in OR)
    vector<AndNfmBounds> lower_bounds_no_cond;
    vector<AndNfmBounds> upper_bounds_no_cond;

    // Map from AND of conditions to AND of lower bounds (MAX of)
    map<BoundConditions, AndNfmBounds> lower_bounds;
    // Map from AND of conditions to AND of upper bounds (MIN of)
    map<BoundConditions, AndNfmBounds> upper_bounds;
    auto& domains = union_dom.get_domains();
    for (size_t i = 0; i < domains.size(); ++i) { // OR of lower/upper bound (IF-ELSE IF-....-ELSE)
        VarDomainBound bound = convert_nfm_domain_to_bound(
            i, domains[i], dim_name, dim_idx, let_assignments);
        //std::cout << "\nBOUND: \n" << bound.to_string() << "\n";

        if (!bound.is_feasible()) { // Empty domain
            //std::cout << "  bound is not feasible\n";
            continue;
        }
        if (bound.conditions.is_always_false()) {
            // The condition is never true
            //std::cout << "  bound is always false\n";
            continue;
        }
        // Lower bound
        if (bound.has_lower_bound()) {
            if (bound.conditions.is_always_true()) {
                if (bound.lower_bounds.size() > 1) {
                    lower_bounds_no_cond.push_back(std::move(bound.lower_bounds));
                } else {
                    const auto& lhs = bound.lower_bounds[0];
                    bool is_subsumed = false;
                    for (auto& rhs : lower_bounds_no_cond) {
                        if (rhs.size() > 1) {
                            continue;
                        }
                        is_subsumed |= do_subsume(lhs, rhs, true);
                    }
                    if (!is_subsumed) {
                        //std::cout << "  Lower bound adding " << lhs.to_string() << " to temp\n";
                        AndNfmBounds temp = {lhs};
                        lower_bounds_no_cond.push_back(std::move(temp));
                    } /*else {
                        std::cout << "  Lower bound adding " << lhs.to_string() << " is SUBSUMED\n";
                    }*/
                }
            } else {
                // Should only appear once (bound with same conditions should have
                // ended up in the same domain in the first place)
                /*std::cout << "  Lower bound adding: \n  COND:\n" << bound.conditions.to_string()
                    << "\n; size: " << lower_bounds.size() << "\n";*/
                bound.conditions.dom_idx = i;
                assert(lower_bounds.find(bound.conditions) == lower_bounds.end());
                lower_bounds.emplace(bound.conditions, bound.lower_bounds);
            }
        }
        // Upper bound
        if (bound.has_upper_bound()) {
            if (bound.conditions.is_always_true()) {
                if (bound.upper_bounds.size() > 1) {
                    upper_bounds_no_cond.push_back(std::move(bound.upper_bounds));
                } else {
                    const auto& lhs = bound.upper_bounds[0];
                    bool is_subsumed = false;
                    for (auto& rhs : upper_bounds_no_cond) {
                        if (rhs.size() > 1) {
                            continue;
                        }
                        is_subsumed |= do_subsume(lhs, rhs, false);
                    }
                    if (!is_subsumed) {
                        //std::cout << "  Upper bound adding " << lhs.to_string() << " to temp\n";
                        AndNfmBounds temp = {lhs};
                        upper_bounds_no_cond.push_back(std::move(temp));
                    } /*else {
                        std::cout << "  Upper bound adding " << lhs.to_string() << " is SUBSUMED\n";
                    }*/
                }
            } else {
                assert(upper_bounds.find(bound.conditions) == upper_bounds.end());
                bound.conditions.dom_idx = i;
                upper_bounds.emplace(bound.conditions, bound.upper_bounds);
            }
        }
    }

    vector<AndNfmBounds> lower_bounds_no_cond_temp;
    for (const auto& bounds : lower_bounds_no_cond) {
        AndNfmBounds temp_no_cond;
        for (const NfmBound& lhs : bounds) {
            size_t size = 0;
            for (auto& iter : lower_bounds) {
                assert(iter.first.dom_idx >= 0 && iter.first.dom_idx < (int)domains.size());
                if (do_subsume(domains[iter.first.dom_idx].get_context_domain(), lhs, iter.second, true)) {
                    size += 1;
                }
            }
            if ((lower_bounds.size() == 0) || (size < lower_bounds.size())) {
                //std::cout << "  no cond lb " << lhs.to_string() << " to temp\n";
                temp_no_cond.push_back(lhs);
            } /*else {
                std::cout << "  no cond lb " << lhs.to_string() << " DROPPING\n";
            }*/
        }
        if (temp_no_cond.size() > 0) {
            if (bounds.size() > 1) { // Can't drop the AND bound if one of them is not dropped
                lower_bounds_no_cond_temp.push_back(bounds);
            } else {
                lower_bounds_no_cond_temp.push_back(std::move(temp_no_cond));
            }
        }
    }
    lower_bounds_no_cond = lower_bounds_no_cond_temp;

    vector<AndNfmBounds> upper_bounds_no_cond_temp;
    for (const auto& bounds : upper_bounds_no_cond) {
        AndNfmBounds temp_no_cond;
        for (const NfmBound& lhs : bounds) {
            size_t size = 0;
            for (auto& iter : upper_bounds) {
                assert(iter.first.dom_idx >= 0 && iter.first.dom_idx < (int)domains.size());
                if (do_subsume(domains[iter.first.dom_idx].get_context_domain(), lhs, iter.second, false)) {
                    size += 1;
                }
            }
            if ((upper_bounds.size() == 0) || (size < upper_bounds.size())) {
                //std::cout << "  no cond ub " << lhs.to_string() << " to temp\n";
                temp_no_cond.push_back(lhs);
            } /*else {
                std::cout << "  no cond ub " << lhs.to_string() << " DROPPING\n";
            }*/
        }
        if (temp_no_cond.size() > 0) {
            if (bounds.size() > 1) { // Can't drop the AND bound if one of them is not dropped
                upper_bounds_no_cond_temp.push_back(bounds);
            } else {
                upper_bounds_no_cond_temp.push_back(std::move(temp_no_cond));
            }
        }
    }
    upper_bounds_no_cond = upper_bounds_no_cond_temp;

    // TODO: Assume that they'are disjoint, i.e you can't have universe (undefined)
    // condition and (M > 2 for example) appear together. Assume that OR of
    // all conditions are the universe (which is always true for Halide interval)

    //std::cout << "\nSTART INTERVAL COMPUTATION\n";
    //std::cout << "Computing interval lower bound\n";
    if (lower_bounds.size() > 0) {
        result.min = convert_to_value(type, union_dom, sym_const_vars, dim_vars, true, lower_bounds);
    }

    //std::cout << "\nComputing interval upper bound\n";
    if (upper_bounds.size() > 0) {
        result.max = convert_to_value(type, union_dom, sym_const_vars, dim_vars, false, upper_bounds);
    }

    for (auto& bounds : lower_bounds_no_cond) {
        assert(bounds.size() > 0);

        Expr lb = convert_nfm_poly_frac_to_halide_expr(
            type, bounds[0].rhs, sym_const_vars, dim_vars, bounds[0].type);
        for (size_t i = 1; i < bounds.size(); ++i) {
            assert((bounds[i].type == EQUAL) || (bounds[i].type == LOWER_BOUND));
            Expr temp = convert_nfm_poly_frac_to_halide_expr(
                type, bounds[i].rhs, sym_const_vars, dim_vars, bounds[i].type);
            lb = max(lb, temp);
        }
        if (result.min.defined()) {
            result.min = Min::make(lb, result.min);
        } else {
            result.min = lb;
        }
    }
    for (auto& bounds : upper_bounds_no_cond) {
        assert(bounds.size() > 0);

        Expr ub = convert_nfm_poly_frac_to_halide_expr(
            type, bounds[0].rhs, sym_const_vars, dim_vars, bounds[0].type);
        for (size_t i = 1; i < bounds.size(); ++i) {
            assert((bounds[i].type == EQUAL) || (bounds[i].type == UPPER_BOUND));
            Expr temp = convert_nfm_poly_frac_to_halide_expr(
                type, bounds[i].rhs, sym_const_vars, dim_vars, bounds[i].type);
            ub = min(ub, temp);
        }
        if (result.max.defined()) {
            result.max = Max::make(ub, result.max);
        } else {
            result.max = ub;
        }
    }

    //debug(0) << "\nresult.min: " << result.min << "\n";
    //debug(0) << "result.max: " << result.max << "\n";
    if ((let_substitutions != NULL) && !let_substitutions->empty()) {
        if (result.min.defined()) {
            result.min = Let::make((*let_substitutions)[let_substitutions->size()-1].first,
                (*let_substitutions)[let_substitutions->size()-1].second, result.min);
            for (int i = let_substitutions->size()-2; i >= 0; --i) {
                result.min = Let::make((*let_substitutions)[i].first, (*let_substitutions)[i].second, result.min);
            }
        }
        if (result.max.defined()) {
            result.max = Let::make((*let_substitutions)[let_substitutions->size()-1].first,
                (*let_substitutions)[let_substitutions->size()-1].second, result.max);
            for (int i = let_substitutions->size()-2; i >= 0; --i) {
                result.max = Let::make((*let_substitutions)[i].first, (*let_substitutions)[i].second, result.max);
            }
        }
    }
    if (expr_substitutions != NULL) {
        result.min = simplify(substitute(*expr_substitutions, result.min));
        result.max = simplify(substitute(*expr_substitutions, result.max));
    } else {
        result.min = simplify(result.min); // NOTE: Simplify sometimes give odd-looking results
        result.max = simplify(result.max);
    }
    //debug(0) << "\nAFTER SUBSTITUTION result.min: " << result.min << "\n";
    //debug(0) << "result.max: " << result.max << "\n";
    return result;
}

// Convert the union_domain into box.
Box convert_nfm_union_domain_to_halide_box(
        Type type, const Nfm::Internal::NfmUnionDomain& p_union_dom,
        const vector<string>& box_dims,
        const vector<string> *let_assignments,
        const map<string, Expr> *expr_substitutions,
        const vector<pair<string, Expr>> *let_substitutions) {
    assert(box_dims.size() > 0);
    if (p_union_dom.is_empty()) {
        return Box();
    }
    if (p_union_dom.is_universe()) {
        Box results;
        results.push_back(Interval(box_dims[0]));
        return results;
    }
    const NfmSpace& coeff_space = p_union_dom.get_coeff_space();
    const NfmSpace& space = p_union_dom.get_space();
    int end_dim_idx = space.get_index(box_dims[box_dims.size()-1]);
    if (end_dim_idx == -1) { // The dimension doesn't exist
        return Box();
    }
    int start_dim_idx = space.get_index(box_dims[0]);
    assert(start_dim_idx != -1);
    assert(end_dim_idx - start_dim_idx + 1 == (int)box_dims.size());

    vector<Expr> sym_const_vars;
    for (size_t i = 0; i < coeff_space.size(); ++i) {
        Expr var = Variable::make(type, coeff_space.get_name(i));
        sym_const_vars.push_back(std::move(var));
    }

    vector<Expr> dim_vars;
    for (size_t i = 0; i < space.size(); ++i) {
        Expr var = Variable::make(type, space.get_name(i));
        dim_vars.push_back(std::move(var));
    }

    // Project out everything after this dim so that it doesn't depend on
    // inner loop vars
    NfmUnionDomain union_dom =
        NfmSolver::nfm_union_domain_eliminate_dims(p_union_dom, (size_t)end_dim_idx+1);
    //std::cout << "Union domain after projection: \n" << union_dom << "\n";

    union_dom = NfmSolver::nfm_union_domain_classify_unknown_coefficient(union_dom, (size_t)end_dim_idx);

    Box results(box_dims.size());

    // Lower and upper bounds without a condition (usually appers in OR)
    vector<vector<AndNfmBounds>> lower_bounds_no_cond_list(box_dims.size());
    vector<vector<AndNfmBounds>> upper_bounds_no_cond_list(box_dims.size());

    // Map from AND of conditions to AND of lower bounds (MAX of)
    vector<map<BoundConditions, AndNfmBounds>> lower_bounds_list(box_dims.size());
    // Map from AND of conditions to AND of upper bounds (MIN of)
    vector<map<BoundConditions, AndNfmBounds>> upper_bounds_list(box_dims.size());

    auto& domains = union_dom.get_domains();
    for (size_t i = 0; i < domains.size(); ++i) { // OR of lower/upper bound (IF-ELSE IF-....-ELSE)
        vector<VarDomainBound> bounds = convert_nfm_domain_to_bound_box_helper(
            i, domains[i], box_dims, start_dim_idx, end_dim_idx, let_assignments);
        assert(bounds.size() == box_dims.size());

        for (int j = box_dims.size()-1; j >= 0; --j) {
            map<BoundConditions, AndNfmBounds>& lower_bounds = lower_bounds_list[j];
            map<BoundConditions, AndNfmBounds>& upper_bounds = upper_bounds_list[j];
            vector<AndNfmBounds>& lower_bounds_no_cond = lower_bounds_no_cond_list[j];
            vector<AndNfmBounds>& upper_bounds_no_cond = upper_bounds_no_cond_list[j];

            VarDomainBound& bound = bounds[j];
            std::cout << "\nBOUND: \n" << bound.to_string() << "\n";
            if (!bound.is_feasible()) { // Empty domain
                continue;
            }
            if (bound.conditions.is_always_false()) {
                // The condition is never true
                continue;
            }
            // Lower bound
            if (bound.has_lower_bound()) {
                if (bound.conditions.is_always_true()) {
                    if (bound.lower_bounds.size() > 1) {
                        lower_bounds_no_cond.push_back(std::move(bound.lower_bounds));
                    } else {
                        const auto& lhs = bound.lower_bounds[0];
                        bool is_subsumed = false;
                        for (auto& rhs : lower_bounds_no_cond) {
                            if (rhs.size() > 1) {
                                continue;
                            }
                            is_subsumed |= do_subsume(lhs, rhs, true);
                        }
                        if (!is_subsumed) {
                            std::cout << "  Lower bound adding " << lhs.to_string() << " to temp\n";
                            AndNfmBounds temp = {lhs};
                            lower_bounds_no_cond.push_back(std::move(temp));
                        } else {
                            std::cout << "  Lower bound adding " << lhs.to_string() << " is SUBSUMED\n";
                        }
                    }
                } else {
                    // Should only appear once (bound with same conditions should have
                    // ended up in the same domain in the first place)
                    std::cout << "  Lower bound adding: \n  COND:\n" << bound.conditions.to_string()
                        << "\n; size: " << lower_bounds.size() << "\n";
                    bound.conditions.dom_idx = i;
                    assert(lower_bounds.find(bound.conditions) == lower_bounds.end());
                    lower_bounds.emplace(bound.conditions, bound.lower_bounds);
                }
            }
            // Upper bound
            if (bound.has_upper_bound()) {
                if (bound.conditions.is_always_true()) {
                    if (bound.upper_bounds.size() > 1) {
                        upper_bounds_no_cond.push_back(std::move(bound.upper_bounds));
                    } else {
                        const auto& lhs = bound.upper_bounds[0];
                        bool is_subsumed = false;
                        for (auto& rhs : upper_bounds_no_cond) {
                            if (rhs.size() > 1) {
                                continue;
                            }
                            is_subsumed |= do_subsume(lhs, rhs, false);
                        }
                        if (!is_subsumed) {
                            std::cout << "  Upper bound adding " << lhs.to_string() << " to temp\n";
                            AndNfmBounds temp = {lhs};
                            upper_bounds_no_cond.push_back(std::move(temp));
                        } else {
                            std::cout << "  Upper bound adding " << lhs.to_string() << " is SUBSUMED\n";
                        }
                    }
                } else {
                    assert(upper_bounds.find(bound.conditions) == upper_bounds.end());
                    bound.conditions.dom_idx = i;
                    upper_bounds.emplace(bound.conditions, bound.upper_bounds);
                }
            }
        }
    }

    for (int j = box_dims.size()-1; j >= 0; --j) {
        map<BoundConditions, AndNfmBounds>& lower_bounds = lower_bounds_list[j];
        map<BoundConditions, AndNfmBounds>& upper_bounds = upper_bounds_list[j];
        vector<AndNfmBounds>& lower_bounds_no_cond = lower_bounds_no_cond_list[j];
        vector<AndNfmBounds>& upper_bounds_no_cond = upper_bounds_no_cond_list[j];

        vector<AndNfmBounds> lower_bounds_no_cond_temp;
        for (const auto& bounds : lower_bounds_no_cond) {
            AndNfmBounds temp_no_cond;
            for (const NfmBound& lhs : bounds) {
                size_t size = 0;
                for (auto& iter : lower_bounds) {
                    assert(iter.first.dom_idx >= 0 && iter.first.dom_idx < (int)domains.size());
                    if (do_subsume(domains[iter.first.dom_idx].get_context_domain(), lhs, iter.second, true)) {
                        size += 1;
                    }
                }
                if ((lower_bounds.size() == 0) || (size < lower_bounds.size())) {
                    std::cout << "  no cond lb " << lhs.to_string() << " to temp\n";
                    temp_no_cond.push_back(lhs);
                } else {
                    std::cout << "  no cond lb " << lhs.to_string() << " DROPPING\n";
                }
            }
            if (temp_no_cond.size() > 0) {
                if (bounds.size() > 1) { // Can't drop the AND bound if one of them is not dropped
                    lower_bounds_no_cond_temp.push_back(bounds);
                } else {
                    lower_bounds_no_cond_temp.push_back(std::move(temp_no_cond));
                }
            }
        }
        lower_bounds_no_cond = lower_bounds_no_cond_temp;

        vector<AndNfmBounds> upper_bounds_no_cond_temp;
        for (const auto& bounds : upper_bounds_no_cond) {
            AndNfmBounds temp_no_cond;
            for (const NfmBound& lhs : bounds) {
                size_t size = 0;
                for (auto& iter : upper_bounds) {
                    assert(iter.first.dom_idx >= 0 && iter.first.dom_idx < (int)domains.size());
                    if (do_subsume(domains[iter.first.dom_idx].get_context_domain(), lhs, iter.second, false)) {
                        size += 1;
                    }
                }
                if ((upper_bounds.size() == 0) || (size < upper_bounds.size())) {
                    std::cout << "  no cond ub " << lhs.to_string() << " to temp\n";
                    temp_no_cond.push_back(lhs);
                } else {
                    std::cout << "  no cond ub " << lhs.to_string() << " DROPPING\n";
                }
            }
            if (temp_no_cond.size() > 0) {
                if (bounds.size() > 1) { // Can't drop the AND bound if one of them is not dropped
                    upper_bounds_no_cond_temp.push_back(bounds);
                } else {
                    upper_bounds_no_cond_temp.push_back(std::move(temp_no_cond));
                }
            }
        }
        upper_bounds_no_cond = upper_bounds_no_cond_temp;

        // TODO: Assume that they'are disjoint, i.e you can't have universe (undefined)
        // condition and (M > 2 for example) appear together. Assume that OR of
        // all conditions are the universe (which is always true for Halide interval)
        Interval& result = results[j];

        std::cout << "\nSTART INTERVAL COMPUTATION\n";
        std::cout << "Computing interval lower bound\n";
        if (lower_bounds.size() > 0) {
            result.min = convert_to_value(type, union_dom, sym_const_vars, dim_vars, true, lower_bounds);
        }

        std::cout << "\nComputing interval upper bound\n";
        if (upper_bounds.size() > 0) {
            result.max = convert_to_value(type, union_dom, sym_const_vars, dim_vars, false, upper_bounds);
        }

        for (auto& bounds : lower_bounds_no_cond) {
            assert(bounds.size() > 0);

            Expr lb = convert_nfm_poly_frac_to_halide_expr(
                type, bounds[0].rhs, sym_const_vars, dim_vars, bounds[0].type);
            for (size_t i = 1; i < bounds.size(); ++i) {
                assert((bounds[i].type == EQUAL) || (bounds[i].type == LOWER_BOUND));
                Expr temp = convert_nfm_poly_frac_to_halide_expr(
                    type, bounds[i].rhs, sym_const_vars, dim_vars, bounds[i].type);
                lb = max(lb, temp);
            }
            if (result.min.defined()) {
                result.min = Min::make(lb, result.min);
            } else {
                result.min = lb;
            }
        }
        for (auto& bounds : upper_bounds_no_cond) {
            assert(bounds.size() > 0);

            Expr ub = convert_nfm_poly_frac_to_halide_expr(
                type, bounds[0].rhs, sym_const_vars, dim_vars, bounds[0].type);
            for (size_t i = 1; i < bounds.size(); ++i) {
                assert((bounds[i].type == EQUAL) || (bounds[i].type == UPPER_BOUND));
                Expr temp = convert_nfm_poly_frac_to_halide_expr(
                    type, bounds[i].rhs, sym_const_vars, dim_vars, bounds[i].type);
                ub = min(ub, temp);
            }
            if (result.max.defined()) {
                result.max = Max::make(ub, result.max);
            } else {
                result.max = ub;
            }
        }

        debug(0) << "\nresult[" << j << "].min: " << result.min << "\n";
        debug(0) << "result[" << j << "].max: " << result.max << "\n";
        if ((let_substitutions != NULL) && !let_substitutions->empty()) {
            if (result.min.defined()) {
                result.min = Let::make((*let_substitutions)[let_substitutions->size()-1].first,
                    (*let_substitutions)[let_substitutions->size()-1].second, result.min);
                for (int i = let_substitutions->size()-2; i >= 0; --i) {
                    result.min = Let::make((*let_substitutions)[i].first, (*let_substitutions)[i].second, result.min);
                }
            }
            if (result.max.defined()) {
                result.max = Let::make((*let_substitutions)[let_substitutions->size()-1].first,
                    (*let_substitutions)[let_substitutions->size()-1].second, result.max);
                for (int i = let_substitutions->size()-2; i >= 0; --i) {
                    result.max = Let::make((*let_substitutions)[i].first, (*let_substitutions)[i].second, result.max);
                }
            }
        }
        if (expr_substitutions != NULL) {
            result.min = simplify(substitute(*expr_substitutions, result.min));
            result.max = simplify(substitute(*expr_substitutions, result.max));
        } else {
            result.min = simplify(result.min); // NOTE: Simplify sometimes give odd-looking results
            result.max = simplify(result.max);
        }
        debug(0) << "\nAFTER SUBSTITUTION result[" << j << "].min: " << result.min << "\n";
        debug(0) << "result[" << j << "].max: " << result.max << "\n";
    }
    return results;
}

//TODO: Add information on loop dimension. Currently, we treat everything as symbolic constants
Interval nfm_simplify_interval(const Interval& interval) {
    Interval result(interval);
    Expr expr;

    // Simplify lower bound
    expr = convert_interval_to_expr_lower_bound(interval);
    //std::cout << "  Simplify lower bound: " << expr << "\n";
    if (expr.defined()) {
        CollectVars collect({interval.var});
        collect.mutate(expr);
        const auto& let_assignments = collect.get_let_assignments();
        map<string, Expr> expr_substitutions;
        vector<pair<string, Expr>> let_substitutions;
        NfmUnionDomain union_dom = convert_halide_expr_to_nfm_union_domain(
            expr, collect.get_sym_consts(), collect.get_dims(), &expr_substitutions, &let_substitutions);
        Interval temp = convert_nfm_union_domain_to_halide_interval(
            Int(32), union_dom, interval.var, &let_assignments, &expr_substitutions, &let_substitutions);
        //std::cout << "    After simplifying lower bound: " << temp.min << "\n";
        result.min = temp.min;
    }

    // Simplify upper bound
    expr = convert_interval_to_expr_upper_bound(interval);
    //std::cout << "  Simplify upper bound: " << expr << "\n";
    if (expr.defined()) {
        CollectVars collect({interval.var});
        collect.mutate(expr);
        const auto& let_assignments = collect.get_let_assignments();
        map<string, Expr> expr_substitutions;
        vector<pair<string, Expr>> let_substitutions;
        NfmUnionDomain union_dom = convert_halide_expr_to_nfm_union_domain(
            expr, collect.get_sym_consts(), collect.get_dims(), &expr_substitutions, &let_substitutions);
        Interval temp = convert_nfm_union_domain_to_halide_interval(
            Int(32), union_dom, interval.var, &let_assignments, &expr_substitutions, &let_substitutions);
        //std::cout << "    After simplifying upper bound: " << temp.max << "\n";
        result.max = temp.max;
    }
    return result;
}

//TODO: Add information on loop dimension. Currently, we treat everything as symbolic constants
//TODO: infer the type from expr
Expr nfm_simplify_expr(const Expr& expr) {
    //std::cout << "\nnfm_simplify_expr: " << expr << "\n";
    if (!expr.defined()) {
        return expr;
    }
    Expr result;
    CollectVars collect;
    collect.mutate(expr);
    const auto& let_assignments = collect.get_let_assignments();
    map<string, Expr> expr_substitutions;
    vector<pair<string, Expr>> let_substitutions;
    NfmUnionDomain union_dom = convert_halide_expr_to_nfm_union_domain(
        expr, collect.get_sym_consts(), collect.get_dims(), &expr_substitutions, &let_substitutions);

    /*debug(0) << "  Union Domain (AFTER SIMPLIFY using NFM) (" << union_dom.get_domains().size() << "): \n";
    union_dom.sort();
    for (const auto& dom : union_dom.get_domains()) {
        debug(0) << "  " << dom << "\n";
        debug(0) << "    Context: " << dom.get_context_domain() << "\n";
    }
    std::cout << "is universe? " << union_dom.is_universe() << "\n";
    std::cout << "is empty? " << union_dom.is_empty() << "\n";*/

    if (union_dom.is_universe()) {
        return make_const(UInt(1), 1);
    } else if (union_dom.is_empty()) {
        return make_const(UInt(1), 0);
    }

    result = convert_nfm_union_domain_to_halide_expr(
        Int(32), union_dom, &let_assignments, &expr_substitutions, &let_substitutions);
    //std::cout << "Result nfm_simplify_expr: " << result << "\n";
    return result;
}

}
}