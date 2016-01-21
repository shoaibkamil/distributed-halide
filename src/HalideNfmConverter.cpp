#include <algorithm>

#include <nfm_solver.h>

#include "HalideNfmConverter.h"

#include "IREquality.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "Module.h"
#include "Simplify.h"

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

// Compute the GCD of two numbers. Always return a non-neg number
int non_neg_gcd(int a, int b) {
    if (a < 0) {
        a = -a;
    }
    if (b < 0) {
        b = -b;
    }
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

int non_neg_lcm(int a, int b) {
    int temp = non_neg_gcd(a, b);
    return temp ? (a / temp * b) : 0;
}

// Compute a vector of integer which element is the max of the two. Save the
// result in exp1
void vector_max_exp(vector<int>& exp1, const vector<int>& exp2) {
    if (exp1.size() != exp2.size()) {
        std::cout << "!!!!vector_max_exp NOT THE SAME; exp1 size: " << exp1.size() << "; exp2 size: " << exp2.size() << "\n";
        for (auto& val : exp1) {
            std::cout << "val exp1: " << val << "\n";
        }
        for (auto& val : exp2) {
            std::cout << "val exp2: " << val << "\n";
        }
    }
    assert(exp1.size() == exp2.size());
    for (size_t i = 0; i < exp1.size(); ++i) {
        assert((exp1[i] >= 0) && (exp2[i] >= 0));
        exp1[i] = std::max(exp1[i], exp2[i]);
    }
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

class HalideNfmConverter : public IRMutator {
protected:
    using IRMutator::visit;

    void visit(const FloatImm *) { error("FloatImm"); }
    void visit(const StringImm *) { error("StringImm"); }
    void visit(const Cast *) { error("Cast"); }
    void visit(const EQ *) { error("EQ"); }
    void visit(const NE *) { error("NE"); }
    void visit(const LT *) { error("LT"); }
    void visit(const LE *) { error("LE"); }
    void visit(const GT *) { error("GT"); }
    void visit(const GE *) { error("GE"); }
    void visit(const And *) { error("And"); }
    void visit(const Or *) { error("Or"); }
    void visit(const Not *op) { error("Not"); }
    void visit(const Select *) { error("Select"); }
    void visit(const Load *) { error("Load"); }
    void visit(const Ramp *) { error("Ramp"); }
    void visit(const Broadcast *) { error("Broadcast"); }
    void visit(const Call *) { error("Call"); }
    void visit(const Let *) { error("Let"); }
    void visit(const LetStmt *) { error("LetStmt"); }
    void visit(const AssertStmt *) { error("AssertStmt"); }
    void visit(const ProducerConsumer *) { error("ProducerConsumer"); }
    void visit(const For *) { error("For"); }
    void visit(const Store *) { error("Store"); }
    void visit(const Provide *) { error("Provide"); }
    void visit(const Allocate *) { error("Allocate"); }
    void visit(const Free *) { error("Free"); }
    void visit(const Realize *) { error("Realize"); }
    void visit(const Block *) { error("Block"); }
    void visit(const IfThenElse *) { error("IfThenElse"); }
    void visit(const Evaluate *) { error("Evaluate"); }

private:
    void error(const std::string& op_name) {
        internal_error << "HalideNfmConverter can't handle " << op_name << "\n";
    }
};

// Return true if it's arithmetic only
class SimpleExpr : public IRVisitor {
public:
    bool is_simple = true;
protected:
    using IRVisitor::visit;

    void visit(const StringImm *) { is_simple = false; }
    void visit(const Cast *) { is_simple = false; }
    void visit(const Min *) { is_simple = false; }
    void visit(const Max *) { is_simple = false; }
    void visit(const EQ *) { is_simple = false; }
    void visit(const NE *) { is_simple = false; }
    void visit(const LT *) { is_simple = false; }
    void visit(const LE *) { is_simple = false; }
    void visit(const GT *) { is_simple = false; }
    void visit(const GE *) { is_simple = false; }
    void visit(const And *) { is_simple = false; }
    void visit(const Or *) { is_simple = false; }
    void visit(const Not *op) { is_simple = false; }
    void visit(const Select *) { is_simple = false; }
    void visit(const Load *) { is_simple = false; }
    void visit(const Ramp *) { is_simple = false; }
    void visit(const Broadcast *) { is_simple = false; }
    void visit(const Call *) { is_simple = false; }
    void visit(const Let *) { is_simple = false; }
    void visit(const LetStmt *) { is_simple = false; }
    void visit(const AssertStmt *) { is_simple = false; }
    void visit(const ProducerConsumer *) { is_simple = false; }
    void visit(const For *) { is_simple = false; }
    void visit(const Store *) { is_simple = false; }
    void visit(const Provide *) { is_simple = false; }
    void visit(const Allocate *) { is_simple = false; }
    void visit(const Free *) { is_simple = false; }
    void visit(const Realize *) { is_simple = false; }
    void visit(const Block *) { is_simple = false; }
    void visit(const IfThenElse *) { is_simple = false; }
    void visit(const Evaluate *) { is_simple = false; }
};

class AddNewVar : public IRMutator {
public:
    AddNewVar() {}

    const vector<string>& get_additional_sym_consts() const {
        return additional_sym_consts;
    }

    const map<string, Expr>& get_expr_substitutions() const {
        return expr_substitutions;
    }

private:
    vector<string> additional_sym_consts;
    map<string, Expr> expr_substitutions;
    map<Expr, string, IRDeepCompare> subs_expr;

    Expr result;

    using IRMutator::visit;

    void visit(const FloatImm *op) {
        // Float has to actually be an integer, e.g. 10.0f
        //debug(0) << "FloatImm: " << op->value << "\n";
        assert(ceilf(op->value) == floorf(op->value));
        expr = (int)op->value;
    }

    void visit(const Cast *op) {
        //debug(0) << "Cast " << op->type << ": " << op->value << "\n";
        if (op->type.is_int() && (op->value.as<IntImm>() != NULL)) {
            expr = mutate(op->value);
        } else {
            assert(op->type.is_int());
            Expr val = Cast::make(op->type, op->value);
            const auto& iter = subs_expr.find(val);
            string var_name;
            if (iter != subs_expr.end()) {
                var_name = iter->second;
            } else {
                var_name = unique_name("_cast");
                additional_sym_consts.push_back(var_name);
                subs_expr.emplace(val, var_name);
                expr_substitutions.emplace(var_name, val);
            }
            Expr var = Variable::make(Int(32), var_name);
            expr = var;
        }
    }

    void visit(const Call *op) {
        //debug(0) << "Call: " << op->name << "\n";
        Expr val = Call::make(op->type, op->name, op->args, op->call_type, op->func,
            op->value_index, op->image, op->param);
        const auto& iter = subs_expr.find(val);
        string var_name;
        if (iter != subs_expr.end()) {
            var_name = iter->second;
        } else {
            var_name = unique_name("_call");
            additional_sym_consts.push_back(var_name);
            subs_expr.emplace(val, var_name);
            expr_substitutions.emplace(var_name, val);
        }
        Expr var = Variable::make(Int(32), var_name);
        expr = var;
    }

    void visit(const Div *op) {
        //debug(0) << "Div: (" << op->a << ")/(" << op->b << "\n";
        Expr val = Div::make(op->a, op->b);
        const auto& iter = subs_expr.find(val);
        string var_name;
        if (iter != subs_expr.end()) {
            var_name = iter->second;
        } else {
            var_name = unique_name("_div");
            additional_sym_consts.push_back(var_name);
            subs_expr.emplace(val, var_name);
            expr_substitutions.emplace(var_name, val);
        }
        Expr var = Variable::make(Int(32), var_name);
        expr = var;
    }

    void visit(const Mod *op) {
        //debug(0) << "Mod: (" << op->a << ")%(" << op->b << "\n";
        Expr val = Mod::make(op->a, op->b);
        const auto& iter = subs_expr.find(val);
        string var_name;
        if (iter != subs_expr.end()) {
            var_name = iter->second;
        } else {
            var_name = unique_name("_mod");
            additional_sym_consts.push_back(var_name);
            subs_expr.emplace(val, var_name);
            expr_substitutions.emplace(var_name, val);
        }
        Expr var = Variable::make(Int(32), var_name);
        expr = var;
    }
};

class PreProcessor : public IRMutator {
protected:
    using IRMutator::visit;

    bool is_simple_arithmetic(Expr expr) {
        SimpleExpr simple;
        expr.accept(&simple);
        //std::cout << "     Expr: " << expr << "; is_simple_arithmetic? " << simple.is_simple << "\n";
        return simple.is_simple;
    }

    void visit(const Add *op) {
        //std::cout << "Add: (" << op->a << ") + (" << op->b << ")\n";
        Expr a = mutate(op->a);
        Expr b = mutate(op->b);
        const Min *min_a = a.as<Min>();
        const Max *max_a = a.as<Max>();
        bool a_simple = is_simple_arithmetic(a);

        const Min *min_b = b.as<Min>();
        const Max *max_b = b.as<Max>();
        bool b_simple = is_simple_arithmetic(b);

        if (min_a && b_simple) {
            expr = Min::make(mutate(min_a->a + b), mutate(min_a->b + b));
        } else if (max_a && b_simple) {
            expr = Max::make(mutate(max_a->a + b), mutate(max_a->b + b));
        } else if (a_simple && min_b) {
            expr = Min::make(mutate(a + min_b->a), mutate(a + min_b->b));
        } else if (a_simple && min_b) {
            expr = Max::make(mutate(a + max_b->a), mutate(a + max_b->b));
        } else {
            IRMutator::visit(op);
        }
        //std::cout << "  Add: (" << op->a << ") + (" << op->b << "); expr: " << expr << "\n";
    }

    void visit(const Sub *op) {
        //std::cout << "Sub: (" << op->a << ") - (" << op->b << ")\n";
        Expr a = mutate(op->a);
        Expr b = mutate(op->b);
        const Min *min_a = a.as<Min>();
        const Max *max_a = a.as<Max>();
        bool a_simple = is_simple_arithmetic(a);

        const Min *min_b = b.as<Min>();
        const Max *max_b = b.as<Max>();
        bool b_simple = is_simple_arithmetic(b);

        if (min_a && b_simple) {
            expr = Min::make(mutate(min_a->a - b), mutate(min_a->b - b));
        } else if (max_a && b_simple) {
            expr = Max::make(mutate(max_a->a - b), mutate(max_a->b - b));
        } else if (a_simple && min_b) {
            expr = Min::make(mutate(a - min_b->a), mutate(a - min_b->b));
        } else if (a_simple && min_b) {
            expr = Max::make(mutate(a - max_b->a), mutate(a - max_b->b));
        } else {
            IRMutator::visit(op);
        }
        //std::cout << "  Sub: (" << op->a << ") - (" << op->b << "); expr: " << expr << "\n";
    }

    void visit(const Mul *op) {
        //std::cout << "Mul: (" << op->a << ") * (" << op->b << ")\n";
        Expr a = mutate(op->a);
        Expr b = mutate(op->b);
        const Min *min_a = a.as<Min>();
        const Max *max_a = a.as<Max>();
        const Min *min_b = b.as<Min>();
        const Max *max_b = b.as<Max>();

        if (min_a) {
            if (is_positive_const(b)) {
                expr = Min::make(mutate(min_a->a*b), mutate(min_a->b*b));
            } else if (is_negative_const(op->b)) {
                expr = Max::make(mutate(min_a->a*b), mutate(min_a->b*b));
            } else {
                IRMutator::visit(op);
            }
        } else if (max_a) {
            if (is_positive_const(b)) {
                expr = Max::make(mutate(max_a->a*b), mutate(max_a->b*b));
            } else if (is_negative_const(b)) {
                expr = Min::make(mutate(max_a->a*b), mutate(max_a->b*b));
            } else {
                IRMutator::visit(op);
            }
        } else if (min_b) {
            if (is_positive_const(a)) {
                expr = Min::make(mutate(a*min_b->a), mutate(a*min_b->b));
            } else if (is_negative_const(op->b)) {
                expr = Max::make(mutate(a*min_b->a), mutate(a*min_b->b));
            } else {
                IRMutator::visit(op);
            }
        } else if (max_b) {
            if (is_positive_const(a)) {
                expr = Max::make(mutate(a*max_b->a), mutate(a*max_b->b));
            } else if (is_negative_const(b)) {
                expr = Min::make(mutate(a*max_b->a), mutate(a*max_b->b));
            } else {
                IRMutator::visit(op);
            }
        } else {
            IRMutator::visit(op);
        }
        //std::cout << "  Mul: (" << op->a << ") * (" << op->b << "); expr: " << expr << "\n";
    }

    void visit(const Div *op) {
        //std::cout << "Div: (" << op->a << ") / (" << op->b << ")\n";
        Expr a = mutate(op->a);
        Expr b = mutate(op->b);
        const Min *min_a = a.as<Min>();
        const Max *max_a = a.as<Max>();
        const Min *min_b = b.as<Min>();
        const Max *max_b = b.as<Max>();

        if (min_a) {
            if (is_positive_const(b)) {
                expr = Min::make(mutate(min_a->a/b), mutate(min_a->b/b));
            } else if (is_negative_const(op->b)) {
                expr = Max::make(mutate(min_a->a/b), mutate(min_a->b/b));
            } else {
                IRMutator::visit(op);
            }
        } else if (max_a) {
            if (is_positive_const(b)) {
                expr = Max::make(mutate(max_a->a/b), mutate(max_a->b/b));
            } else if (is_negative_const(b)) {
                expr = Min::make(mutate(max_a->a/b), mutate(max_a->b/b));
            } else {
                IRMutator::visit(op);
            }
        } else if (min_b) {
            if (is_positive_const(a)) {
                expr = Min::make(mutate(a/min_b->a), mutate(a/min_b->b));
            } else if (is_negative_const(op->b)) {
                expr = Max::make(mutate(a/min_b->a), mutate(a/min_b->b));
            } else {
                IRMutator::visit(op);
            }
        } else if (max_b) {
            if (is_positive_const(a)) {
                expr = Max::make(mutate(a/max_b->a), mutate(a/max_b->b));
            } else if (is_negative_const(b)) {
                expr = Min::make(mutate(a/max_b->a), mutate(a/max_b->b));
            } else {
                IRMutator::visit(op);
            }
        } else {
            IRMutator::visit(op);
        }
        //std::cout << "  Div: (" << op->a << ") / (" << op->b << "); expr: " << expr << "\n";
    }
};

// Transform expression into system of inequalities/equalities
class ConvertToIneqs : public HalideNfmConverter {
public:
    ConvertToIneqs() {}

    struct ConditionVal {
        ConditionVal() {}
        ConditionVal(Expr c, Expr v) : cond(c), val(v) {}
        Expr cond;
        Expr val;
    };

    Expr get_result() {
        if (result.defined()) {
            return result;
        }
        result = expr;
        for (auto& cst : additional_constraints) {
            result = result && cst;
        }
        return result;
    }

    const vector<pair<string, Expr>>& get_let_substitutions() const {
        return let_substitutions;
    }

private:
    vector<Expr> additional_constraints;
    vector<ConditionVal> select_vals; // Disjuction of values (ORs)
    vector<pair<string, Expr>> let_substitutions;

    Expr result;

    using HalideNfmConverter::visit;

    template <typename T>
    void visit_helper_arithmetic(const T *op) {
        //debug(0) << "visit_helper_arithmetic: " << op->a << " and " << op->b << "\n";
        vector<ConditionVal> a_cond_vals;
        vector<ConditionVal> b_cond_vals;
        Expr a = mutate(op->a);
        a_cond_vals.swap(select_vals);
        Expr b = mutate(op->b);
        b_cond_vals.swap(select_vals);

        bool select_a = a_cond_vals.size() > 0;
        bool select_b = b_cond_vals.size() > 0;
        if (select_a && select_b) {
            for (auto cond_val_a : a_cond_vals) {
                for (auto cond_val_b : b_cond_vals) {
                    select_vals.push_back(
                        ConditionVal(And::make(cond_val_a.cond, cond_val_b.cond),
                                     T::make(cond_val_a.val, cond_val_b.val)));
                }
            }
        } else if (select_a) {
            for (auto cond_val_a : a_cond_vals) {
                select_vals.push_back(
                    ConditionVal(cond_val_a.cond, T::make(cond_val_a.val, b)));
            }
        } else if (select_b) {
            for (auto cond_val_b : b_cond_vals) {
                select_vals.push_back(
                    ConditionVal(cond_val_b.cond, T::make(a, cond_val_b.val)));
            }
        } else {
            expr = T::make(a, b);
        }
        if (select_a || select_b) {
            string var_name = unique_name("_dummy_arith");
            Expr var = Variable::make(select_vals[0].val.type(), var_name);
            assert(select_vals.size() > 0);
            if (select_vals.size() == 1) {
                expr = And::make(select_vals[0].cond, EQ::make(var, select_vals[0].val));
                return;
            }
            expr = Or::make(And::make(select_vals[0].cond, EQ::make(var, select_vals[0].val)),
                            And::make(select_vals[1].cond, EQ::make(var, select_vals[1].val)));
            for (size_t i = 2; i < select_vals.size(); ++i) {
                expr = Or::make(expr, And::make(select_vals[i].cond, EQ::make(var, select_vals[i].val)));
            }
        }
    }

    template <typename T>
    void visit_helper_ineq(const T *op) {
        //debug(0) << "visit_helper_ineq: " << op->a << " and " << op->b << "\n";
        select_vals.clear();
        vector<ConditionVal> a_cond_vals;
        vector<ConditionVal> b_cond_vals;
        Expr a = mutate(op->a);
        a_cond_vals.swap(select_vals);
        Expr b = mutate(op->b);
        b_cond_vals.swap(select_vals);

        bool select_a = a_cond_vals.size() > 0;
        bool select_b = b_cond_vals.size() > 0;
        if (select_a && select_b) {
            vector<ConditionVal> result;
            for (auto cond_val_a : a_cond_vals) {
                for (auto cond_val_b : b_cond_vals) {
                    result.push_back(
                        ConditionVal(And::make(cond_val_a.cond, cond_val_b.cond),
                                     T::make(cond_val_a.val-cond_val_b.val, 0)));
                }
            }
            assert(result.size() > 0);
            if (result.size() == 1) {
                expr = And::make(result[0].cond, result[0].val);
                return;
            }
            expr = Or::make(And::make(result[0].cond, result[0].val),
                            And::make(result[1].cond, result[1].val));
            for (size_t i = 2; i < result.size(); ++i) {
                expr = Or::make(expr, And::make(result[i].cond, result[i].val));
            }
        } else if (select_a) {
            assert(a_cond_vals.size() > 0);
            if (a_cond_vals.size() == 1) {
                expr = And::make(a_cond_vals[0].cond, T::make(a_cond_vals[0].val-b, 0));
                return;
            }
            expr = Or::make(And::make(a_cond_vals[0].cond, T::make(a_cond_vals[0].val-b, 0)),
                            And::make(a_cond_vals[1].cond, T::make(a_cond_vals[1].val-b, 0)));
            for (size_t i = 2; i < a_cond_vals.size(); ++i) {
                expr = Or::make(expr, And::make(a_cond_vals[i].cond, T::make(a_cond_vals[i].val-b, 0)));
            }
        } else if (select_b) {
            assert(b_cond_vals.size() > 0);
            if (b_cond_vals.size() == 1) {
                expr = And::make(b_cond_vals[0].cond, T::make(a-b_cond_vals[0].val, 0));
                return;
            }
            expr = Or::make(And::make(b_cond_vals[0].cond, T::make(a-b_cond_vals[0].val, 0)),
                            And::make(b_cond_vals[1].cond, T::make(a-b_cond_vals[1].val, 0)));
            for (size_t i = 2; i < b_cond_vals.size(); ++i) {
                expr = Or::make(expr, And::make(b_cond_vals[i].cond, T::make(a-b_cond_vals[i].val, 0)));
            }
        } else {
            expr = T::make(a-b, 0);
        }
    }

    template <typename T>
    void visit_helper_minmax(const T *op, bool is_min) {
        //debug(0) << "visit_helper_minmax: " << op->a << " and " << op->b << "\n";
        select_vals.clear();
        vector<ConditionVal> a_cond_vals;
        vector<ConditionVal> b_cond_vals;
        Expr a = mutate(op->a);
        a_cond_vals.swap(select_vals);
        Expr b = mutate(op->b);
        b_cond_vals.swap(select_vals);

        string var_name = unique_name("_dummy_minmax");
        Expr var = Variable::make(op->a.type(), var_name);
        assert(var.type() == op->b.type());

        bool select_a = a_cond_vals.size() > 0;
        bool select_b = b_cond_vals.size() > 0;
        vector<ConditionVal> result;
        Expr cond1, cond2;
        if (select_a && select_b) {
            for (auto cond_val_a : a_cond_vals) {
                for (auto cond_val_b : b_cond_vals) {
                    if (is_min) {
                        cond1 = mutate(LE::make(cond_val_a.val, cond_val_b.val));
                        cond2 = mutate(GT::make(cond_val_a.val, cond_val_b.val));
                    } else {
                        cond1 = mutate(GE::make(cond_val_a.val, cond_val_b.val));
                        cond2 = mutate(LT::make(cond_val_a.val, cond_val_b.val));
                    }
                    result.push_back(ConditionVal(
                        And::make(cond1, And::make(cond_val_a.cond, cond_val_b.cond)),
                        cond_val_a.val));
                    result.push_back(ConditionVal(
                        And::make(cond2, And::make(cond_val_a.cond, cond_val_b.cond)),
                        cond_val_b.val));
                }
            }
            assert(result.size() > 0);
            if (result.size() == 1) {
                expr = And::make(result[0].cond, EQ::make(var, result[0].val));
            } else {
                expr = Or::make(And::make(result[0].cond, EQ::make(var, result[0].val)),
                                And::make(result[1].cond, EQ::make(var, result[1].val)));
                for (size_t i = 2; i < result.size(); ++i) {
                    expr = Or::make(expr, And::make(result[i].cond, EQ::make(var, result[i].val)));
                }
            }
        } else if (select_a) {
            for (auto cond_val : a_cond_vals) {
                if (is_min) {
                    cond1 = mutate(LE::make(cond_val.val, b));
                    cond2 = mutate(GT::make(cond_val.val, b));
                } else {
                    cond1 = mutate(GE::make(cond_val.val, b));
                    cond2 = mutate(LT::make(cond_val.val, b));
                }
                result.push_back(
                    ConditionVal(And::make(cond1, cond_val.cond), cond_val.val));
                result.push_back(
                    ConditionVal(And::make(cond2, cond_val.cond), b));
            }
            assert(result.size() > 0);
            if (result.size() == 1) {
                expr = And::make(result[0].cond, EQ::make(var, result[0].val));
            } else {
                expr = Or::make(And::make(result[0].cond, EQ::make(var, result[0].val)),
                                And::make(result[1].cond, EQ::make(var, result[1].val)));
                for (size_t i = 2; i < result.size(); ++i) {
                    expr = Or::make(expr, And::make(result[i].cond, EQ::make(var, result[i].val)));
                }
            }
        } else if (select_b) {
            for (auto cond_val : b_cond_vals) {
                if (is_min) {
                    cond1 = mutate(LE::make(a, cond_val.val));
                    cond2 = mutate(GT::make(a, cond_val.val));
                } else {
                    cond1 = mutate(GE::make(a, cond_val.val));
                    cond2 = mutate(LT::make(a, cond_val.val));
                }
                result.push_back(
                    ConditionVal(And::make(cond1, cond_val.cond), a));
                result.push_back(
                    ConditionVal(And::make(cond2, cond_val.cond), cond_val.val));
            }
            assert(result.size() > 0);
            if (result.size() == 1) {
                expr = And::make(result[0].cond, EQ::make(var, result[0].val));
            } else {
                expr = Or::make(And::make(result[0].cond, EQ::make(var, result[0].val)),
                                And::make(result[1].cond, EQ::make(var, result[1].val)));
                for (size_t i = 2; i < result.size(); ++i) {
                    expr = Or::make(expr, And::make(result[i].cond, EQ::make(var, result[i].val)));
                }
            }
        } else {
            if (is_min) {
                cond1 = mutate(LE::make(a, b));
                cond2 = mutate(GT::make(a, b));
            } else {
                cond1 = mutate(GE::make(a, b));
                cond2 = mutate(LT::make(a, b));
            }
            result.push_back(ConditionVal(cond1, a));
            result.push_back(ConditionVal(cond2, b));
            Expr conj1 = And::make(cond1, EQ::make(var, a));
            Expr conj2 = And::make(cond2, EQ::make(var, b));
            expr = Or::make(conj1, conj2);
        }
        select_vals.swap(result);
    }

    void visit(const Variable* op) {
        // Sometime we have the following case: select(x, 10, 2) where x is a
        // boolean. Need to convert the condition into a equality (select(x==1, 10, 2))
        // when converting into NFM
        if (op->type.is_bool()) {
            expr = (Variable::make(Int(32), op->name) == 1);
        } else {
            HalideNfmConverter::visit(op);
        }
    }

    void visit(const FloatImm *op) {
        // Float has to actually be an integer, e.g. 10.0f
        //debug(0) << "FloatImm: " << op->value << "\n";
        internal_error << "ConvertToNfmStructs can't handle FloatImm\n";
    }

    void visit(const Cast *op) {
        //debug(0) << "Cast " << op->type << ": " << op->value << "\n";
        internal_error << "ConvertToNfmStructs can't handle Cast\n";
    }

    // Only handle call to ceil_f32 or floor_f32 (single argument). We do
    // nothing about the ceil or floor at this stage. We'll add the ceil/floor
    // back during the conversion from nfm to halide.
    void visit(const Call *op) {
        /*user_assert((op->name == "ceil_f32") || (op->name == "floor_f32"))
            << "ConvertToIneqs only handle ceil_f32 or floor_f32: " << op->name << "\n";
        user_assert(op->args.size() == 1) << "op->args.size(): " << op->args.size() << "\n";
        expr = mutate(op->args[0]);*/

        //debug(0) << "Call: " << op->name << "\n";
        internal_error << "ConvertToNfmStructs can't handle Call\n";
    }

    template <typename T>
    Expr convert_select_helper(const T *op, const Select *select_a, const Select *select_b) {
        Expr new_expr;
        if (select_a && select_b) {
            new_expr = select(select_a->condition,
                          select(select_b->condition,
                                 T::make(select_a->true_value, select_b->true_value),
                                 T::make(select_a->true_value, select_b->false_value)),
                          select(select_b->condition,
                                 T::make(select_a->false_value, select_b->true_value),
                                 T::make(select_a->false_value, select_b->false_value)));
        } else if (select_a) {
            new_expr = select(select_a->condition,
                          T::make(select_a->true_value, op->b),
                          T::make(select_a->false_value, op->b));
        } else if (select_b) {
            new_expr = select(select_b->condition,
                          T::make(op->a, select_b->true_value),
                          T::make(op->a, select_b->false_value));
        }
        return new_expr;
    }

    void visit(const Add *op) {
        //debug(0) << "Add: (" << op->a << ")+(" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        visit_helper_arithmetic(op);
    }

    void visit(const Sub *op) {
        //debug(0) << "Sub: (" << op->a << ")-(" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        visit_helper_arithmetic(op);
    }

    void visit(const Mul *op) {
        //debug(0) << "Mul: (" << op->a << ")*(" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        visit_helper_arithmetic(op);
    }

    void visit(const Div *op) {
        //debug(0) << "Div: (" << op->a << ")/(" << op->b << ")\n";
        /*Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }*/
        // Normalize division, e.g convert w >= y/c for example into the following:
        // (w >= t) and (c*t <= y <= c*t + c - 1). NOTE: c has to be symbolic constant
        // and positive. Division is equivalent to floor function
        // TODO: Expand this to handle the case when c is not symbolic constant and
        // is not positive
        internal_error << "ConvertToNfmStructs can't handle Div\n";
    }

    void visit(const Mod *op) {
        //TODO: handle modulus (q = floor(op->a/op->b), result = op->a-q*op->b)
        internal_error << "ConvertToNfmStructs can't handle Mod\n";
    }

    void visit(const Min *op) {
        //debug(0) << "Min: (" << op->a << ") and (" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        visit_helper_minmax(op, true);
        //debug(0) << "Min: (" << op->a << ") and (" << op->b << "); \n  RESULT: " << expr << "\n";
    }

    void visit(const Max *op) {
        //debug(0) << "Max: (" << op->a << ") and (" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        visit_helper_minmax(op, false);
        //debug(0) << "Max: (" << op->a << ") and (" << op->b << "); \n  RESULT: " << expr << "\n";
    }

    void visit(const EQ *op) {
        // Convert: a == b into (a-b == 0)
        //debug(0) << "Eq: (" << op->a << ") = (" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        const Let *let_a = op->a.as<Let>();
        const Let *let_b = op->b.as<Let>();
        if (let_a && let_b) {
            expr = mutate(Let::make(let_a->name, let_a->value,
                                    Let::make(let_b->name, let_b->value,
                                              EQ::make(let_a->body, let_b->body))));
        } else if (let_a) {
            expr = mutate(Let::make(let_a->name, let_a->value, EQ::make(let_a->body, op->b)));
        } else if (let_b) {
            expr = mutate(Let::make(let_b->name, let_b->value, EQ::make(op->a, let_b->body)));
        } else {
            visit_helper_ineq(op);
        }
    }

    void visit(const NE *op) {
        // Convert: a != b into a < b or a > b
        //debug(0) << "NE: (" << op->a << ") != (" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        Expr cons1 = mutate(LT::make(op->a, op->b));
        Expr cons2 = mutate(GT::make(op->a, op->b));
        expr = mutate(Or::make(cons1, cons2));
    }

    void visit(const LT *op) {
        // Convert: a < b into a <= b - 1 into b - a - 1 >= 0
        //debug(0) << "LT: (" << op->a << ") < (" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        const Max *max_a = op->a.as<Max>();
        const Min *min_a = op->a.as<Min>();
        const Max *max_b = op->b.as<Max>();
        const Min *min_b = op->b.as<Min>();
        if (max_a) {
            expr = mutate(And::make(max_a->a < op->b, max_a->b < op->b));
        } else if (min_b) {
            expr = mutate(And::make(op->a < min_b->a, op->a < min_b->b));
        } else if (min_a) {
            expr = mutate(Or::make(min_a->a < op->b, min_a->b < op->b));
        } else if (max_b) {
            expr = mutate(Or::make(op->a < max_b->a, op->a < max_b->b));
        } else {
            expr = mutate(GE::make(op->b, op->a+1));
        }
    }

    void visit(const LE *op) {
        // Convert: a <= b into b - a >= 0
        //debug(0) << "LE: (" << op->a << ") <= (" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        const Max *max_a = op->a.as<Max>();
        const Min *min_a = op->a.as<Min>();
        const Max *max_b = op->b.as<Max>();
        const Min *min_b = op->b.as<Min>();
        if (max_a) {
            expr = mutate(And::make(max_a->a <= op->b, max_a->b <= op->b));
        } else if (min_b) {
            expr = mutate(And::make(op->a <= min_b->a, op->a <= min_b->b));
        } else if (min_a) {
            expr = mutate(Or::make(min_a->a <= op->b, min_a->b <= op->b));
        } else if (max_b) {
            expr = mutate(Or::make(op->a <= max_b->a, op->a <= max_b->b));
        } else {
            expr = mutate(GE::make(op->b, op->a));
        }
        //debug(0) << "LE: (" << op->a << ") <= (" << op->b << "); \n  RESULT: " << expr << "\n";
    }

    void visit(const GT *op) {
        // Convert: a > b into a >= b + 1 into a - b - 1 >= 0
        //debug(0) << "GT: (" << op->a << ") > (" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        const Max *max_a = op->a.as<Max>();
        const Min *min_a = op->a.as<Min>();
        const Max *max_b = op->b.as<Max>();
        const Min *min_b = op->b.as<Min>();
        if (min_a) {
            expr = mutate(And::make(min_a->a > op->b, min_a->b > op->b));
        } else if (max_b) {
            expr = mutate(And::make(op->a > max_b->a, op->a > max_b->b));
        } else if (max_a) {
            expr = mutate(Or::make(max_a->a > op->b, max_a->b > op->b));
        } else if (min_b) {
            expr = mutate(Or::make(op->a > min_b->a, op->a > min_b->b));
        } else {
            expr = mutate(GE::make(op->a-1, op->b));
        }
    }

    void visit(const GE *op) {
        // Convert: a >= b into a - b >= 0
        //debug(0) << "GE: (" << op->a << ") >= (" << op->b << ")\n";
        Expr new_expr = convert_select_helper(op, op->a.as<Select>(), op->b.as<Select>());
        if (new_expr.defined()) {
            expr = mutate(new_expr);
            return;
        }
        const Let *let_a = op->a.as<Let>();
        const Let *let_b = op->b.as<Let>();
        if (let_a && let_b) {
            expr = mutate(Let::make(let_a->name, let_a->value,
                                    Let::make(let_b->name, let_b->value,
                                              GE::make(let_a->body, let_b->body))));
        } else if (let_a) {
            expr = mutate(Let::make(let_a->name, let_a->value, GE::make(let_a->body, op->b)));
        } else if (let_b) {
            expr = mutate(Let::make(let_b->name, let_b->value, GE::make(op->a, let_b->body)));
        } else {
            const Max *max_a = op->a.as<Max>();
            const Min *min_a = op->a.as<Min>();
            const Max *max_b = op->b.as<Max>();
            const Min *min_b = op->b.as<Min>();
            const Add *add_a = op->a.as<Add>();
            const Add *add_b = op->b.as<Add>();
            const Sub *sub_a = op->a.as<Sub>();
            const Sub *sub_b = op->b.as<Sub>();

            if (min_a) {
                expr = mutate(And::make(min_a->a >= op->b, min_a->b >= op->b));
            } else if (max_b) {
                expr = mutate(And::make(op->a >= max_b->a, op->a >= max_b->b));
            } else if (max_a) {
                expr = mutate(Or::make(max_a->a >= op->b, max_a->b >= op->b));
            } else if (min_b) {
                expr = mutate(Or::make(op->a >= min_b->a, op->a >= min_b->b));
            } else if (add_a) {
                const Max *add_a_max_a = add_a->a.as<Max>();
                const Min *add_a_min_a = add_a->a.as<Min>();
                const Max *add_a_max_b = add_a->b.as<Max>();
                const Min *add_a_min_b = add_a->b.as<Min>();
                if (add_a_max_a) {
                    expr = mutate(GE::make(add_a_max_a, op->b - add_a->b));
                } else if (add_a_min_a) {
                    expr = mutate(GE::make(add_a_min_a, op->b - add_a->b));
                } else if (add_a_max_b) {
                    expr = mutate(GE::make(add_a_max_b, op->b - add_a->a));
                } else if (add_a_min_b) {
                    expr = mutate(GE::make(add_a_min_b, op->b - add_a->a));
                } else {
                    visit_helper_ineq(op);
                }
            } else if (sub_a) {
                const Max *sub_a_max_a = sub_a->a.as<Max>();
                const Min *sub_a_min_a = sub_a->a.as<Min>();
                const Max *sub_a_max_b = sub_a->b.as<Max>();
                const Min *sub_a_min_b = sub_a->b.as<Min>();
                if (sub_a_max_a) {
                    expr = mutate(GE::make(sub_a_max_a, op->b + sub_a->b));
                } else if (sub_a_min_a) {
                    expr = mutate(GE::make(sub_a_min_a, op->b + sub_a->b));
                } else if (sub_a_max_b) {
                    expr = mutate(GE::make(sub_a_max_b, op->b + sub_a->a));
                } else if (sub_a_min_b) {
                    expr = mutate(GE::make(sub_a_min_b, op->b + sub_a->a));
                } else {
                    visit_helper_ineq(op);
                }
            } else if (add_b) {
                const Max *add_b_max_a = add_b->a.as<Max>();
                const Min *add_b_min_a = add_b->a.as<Min>();
                const Max *add_b_max_b = add_b->b.as<Max>();
                const Min *add_b_min_b = add_b->b.as<Min>();
                if (add_b_max_a) {
                    expr = mutate(GE::make(op->a - add_b->b, add_b_max_a));
                } else if (add_b_min_a) {
                    expr = mutate(GE::make(op->a - add_b->b, add_b_min_a));
                } else if (add_b_max_b) {
                    expr = mutate(GE::make(op->a - add_b->a, add_b_max_b));
                } else if (add_b_min_b) {
                    expr = mutate(GE::make(op->a - add_b->a, add_b_min_b));
                } else {
                    visit_helper_ineq(op);
                }
            } else if (sub_b) {
                const Max *sub_b_max_a = sub_b->a.as<Max>();
                const Min *sub_b_min_a = sub_b->a.as<Min>();
                const Max *sub_b_max_b = sub_b->b.as<Max>();
                const Min *sub_b_min_b = sub_b->b.as<Min>();
                if (sub_b_max_a) {
                    expr = mutate(GE::make(op->a + sub_b->b, sub_b_max_a));
                } else if (sub_b_min_a) {
                    expr = mutate(GE::make(op->a + sub_b->b, sub_b_min_a));
                } else if (sub_b_max_b) {
                    expr = mutate(GE::make(op->a + sub_b->a, sub_b_max_b));
                } else if (sub_b_min_b) {
                    expr = mutate(GE::make(op->a + sub_b->a, sub_b_min_b));
                } else {
                    visit_helper_ineq(op);
                }
            } else {
                visit_helper_ineq(op);
            }
        }
        //debug(0) << "GE: (" << op->a << ") >= (" << op->b << "); \n  RESULT: " << expr << "\n";
    }

    void visit(const And *op) {
        //debug(0) << "And: (" << op->a << ") and (" << op->b << ")\n";
        Expr a = mutate(op->a);
        Expr b = mutate(op->b);
        expr = And::make(a, b);
    }

    void visit(const Or *op) {
        //debug(0) << "Or: (" << op->a << ") or (" << op->b << ")\n";
        Expr a = mutate(op->a);
        Expr b = mutate(op->b);
        expr = Or::make(a, b);
    }

    void visit(const Not *op) {
        //debug(0) << "Not: !(" << op->a << ")\n";
        Expr a = mutate(op->a);
        const EQ *eq_a = a.as<EQ>();
        const NE *ne_a = a.as<NE>();
        const LT *lt_a = a.as<LT>();
        const LE *le_a = a.as<LE>();
        const GT *gt_a = a.as<GT>();
        const GE *ge_a = a.as<GE>();
        const And *ands_a = a.as<And>();
        const Or *or_a = a.as<Or>();
        const Not *not_a = a.as<Not>();
        if (eq_a) {
            // !(a = b) -> (a != b)
            expr = mutate(NE::make(eq_a->a, eq_a->b));
        } else if (ne_a) {
            // !(a != b) -> (a = b)
            expr = mutate(EQ::make(ne_a->a, ne_a->b));
        } else if (lt_a) {
            // !(a < b) -> (a >= b)
            expr = mutate(GE::make(lt_a->a, lt_a->b));
        } else if (le_a) {
            // !(a <= b) -> (a > b)
            expr = mutate(GT::make(le_a->a, le_a->b));
        } else if (gt_a) {
            // !(a > b) -> (a <= b)
            expr = mutate(LE::make(gt_a->a, gt_a->b));
        } else if (ge_a) {
            // !(a >= b) -> (a <= b)
            expr = mutate(LT::make(ge_a->a, ge_a->b));
        } else if (ands_a) {
            // !(a && b) -> (!a || !b)
            expr = mutate(Or::make(!ands_a->a, !ands_a->b));
        } else if (or_a) {
            // !(a || b) -> (!a && !b)
            expr = mutate(And::make(!or_a->a, !or_a->b));
        } else if (not_a) {
            // !(!a) -> a
            expr = mutate(not_a->a);
        } else {
            expr = Not::make(a);
        }
        //debug(0) << "  Not result: " << (expr) << "\n";
    }

    void visit(const Select *op) {
        /*debug(0) << "Select: (" << op->condition << ") : (" << op->true_value
                 << ") ? (" << op->false_value << ")\n";*/
        select_vals.clear();
        vector<ConditionVal> true_cond_val;
        vector<ConditionVal> false_cond_val;

        assert(!op->condition.as<Select>()); // Condition shouldn't have had been a Select
        assert(op->true_value.type() == op->false_value.type());
        string var_name = unique_name("_dummy");
        Expr var = Variable::make(op->true_value.type(), var_name);
        assert(var.type() == op->false_value.type());

        Expr cond = mutate(op->condition);
        //debug(0) << "  cond: " << simplify(cond) << "\n";
        Expr not_cond = mutate(!op->condition);
        //debug(0) << "  not_cond: " << simplify(not_cond) << "\n";
        Expr true_value = mutate(op->true_value);
        true_cond_val.swap(select_vals);
        //debug(0) << "  true_value: (" << true_cond_val.size() << "): " << simplify(true_value) << "\n";
        Expr false_value = mutate(op->false_value);
        false_cond_val.swap(select_vals);
        //debug(0) << "  false_value: (" << false_cond_val.size() << "): " << simplify(false_value) << "\n";

        bool select_a = true_cond_val.size();   // True value
        bool select_b = false_cond_val.size();  // False value
        if (select_a && select_b) {
            assert(select_vals.size() == 0);
            for (auto cond_val : true_cond_val) {
                select_vals.push_back(
                  ConditionVal(And::make(cond, cond_val.cond), cond_val.val));
            }
            for (auto cond_val : false_cond_val) {
                select_vals.push_back(
                  ConditionVal(And::make(not_cond, cond_val.cond), cond_val.val));
            }
        } else if (select_a) {
            assert(select_vals.size() == 0);
            assert(false_cond_val.size() == 0);
            select_vals.insert(select_vals.end(), ConditionVal(not_cond, false_value));
            for (auto cond_val : true_cond_val) {
                select_vals.push_back(
                  ConditionVal(And::make(cond, cond_val.cond), cond_val.val));
            }
        } else if (select_b) {
            assert(select_vals.size() == 0);
            assert(true_cond_val.size() == 0);
            select_vals.insert(select_vals.end(), ConditionVal(cond, true_value));
            for (auto cond_val : false_cond_val) {
                select_vals.push_back(
                  ConditionVal(And::make(not_cond, cond_val.cond), cond_val.val));
            }
        } else {
            select_vals.push_back(ConditionVal(cond, true_value));
            select_vals.push_back(ConditionVal(not_cond, false_value));
        }

        assert(select_vals.size() >= 2);
        Expr true_expr = select_vals[0].val;
        Expr false_expr = select_vals[1].val;
        if (!true_expr.type().is_bool()) {
            true_expr = EQ::make(var, true_expr);
        }
        if (!false_expr.type().is_bool()) {
            false_expr = EQ::make(var, false_expr);
        }
        expr = Or::make(And::make(select_vals[0].cond, true_expr),
                        And::make(select_vals[1].cond, false_expr));
        for (size_t i = 2; i < select_vals.size(); ++i) {
            Expr val = select_vals[i].val;
            if (!val.type().is_bool()) {
                val = EQ::make(var, val);
            }
            expr = Or::make(expr, And::make(select_vals[i].cond, val));
        }
        //debug(0) << "  RESULT SELECT: " << simplify(expr) << "\n";
    }

    void visit(const Let *op) {
        // Convert op->name = op->value into op->name - op->value = 0
        //debug(0) << "Let: let(" << op->name << " = " << op->value << "); " << op->body << "\n";
        // TODO: need to simplify this as well. Might need to convert the 'let'
        // into nfm and simplify it separately.
        /*const Expr& val = op->value;
        const auto& iter = subs_expr.find(val);
        string var_name;
        if (iter != subs_expr.end()) {
            var_name = iter->second;
        } else {
            //var_name = unique_name("_let");
            var_name = op->name;
            additional_sym_consts.push_back(var_name);
            subs_expr.emplace(val, var_name);
            expr_substitutions.emplace(var_name, val);
        }
        Expr eq = EQ::make(Variable::make(op->value.type(), op->name),
                           Variable::make(Int(32), var_name));
        Expr body = mutate(op->body);
        expr = mutate(And::make(eq, body));*/

        let_substitutions.push_back(std::make_pair(op->name, op->value));
        expr = mutate(op->body);
    }
};


class ConvertToDNF : public IRMutator {
public:
    ConvertToDNF() {}
private:
    using IRMutator::visit;

    void visit(const And *op) {
        //debug(0) << "And: (" << op->a << ") and (" << op->b << ")\n";
        Expr a = mutate(op->a);
        Expr b = mutate(op->b);

        const Or *or_a = a.as<Or>();
        const Or *or_b = b.as<Or>();
        if (or_a && or_b) {
            expr = mutate(Or::make(And::make(or_a->a, or_b->a),
                                   Or::make(And::make(or_a->b, or_b->a),
                                            Or::make(And::make(or_a->a, or_b->b),
                                                     And::make(or_a->b, or_b->b)))));
        } else if (or_a) {
            expr = mutate(Or::make(And::make(or_a->a, b), And::make(or_a->b, b)));
        } else if (or_b) {
            expr = mutate(Or::make(And::make(a, or_b->a), And::make(a, or_b->b)));
        } else {
            expr = And::make(a, b);
        }
        //debug(0) << "  And Result: " << expr << "\n";
    }

    void visit(const Or *op) {
        //debug(0) << "Or: (" << op->a << ") and (" << op->b << ")\n";
        Expr a = mutate(op->a);
        Expr b = mutate(op->b);
        expr = Or::make(a, b);
        //debug(0) << "  Or Result: " << expr << "\n";
    }
};

// Assume everything is already in DNF (OR of AND). Does not handle the case
// when OR is inside an AND (AND (A OR B))
class SplitAnds : public HalideNfmConverter {
public:
    vector<Expr> result;
    SplitAnds() {}
private:
    using HalideNfmConverter::visit;

    void visit(const EQ *op) {
        result.push_back(EQ::make(op->a, op->b));
    }

    void visit(const NE *op) {
        result.push_back(NE::make(op->a, op->b));
    }

    void visit(const LT *op) {
        result.push_back(LT::make(op->a, op->b));
    }

    void visit(const LE *op) {
        result.push_back(LE::make(op->a, op->b));
    }

    void visit(const GT *op) {
        result.push_back(GT::make(op->a, op->b));
    }

    void visit(const GE *op) {
        result.push_back(GE::make(op->a, op->b));
    }

    void visit(const And *op) {
        mutate(op->a);
        mutate(op->b);
    }

    void visit(const Or *op) {
        result.push_back(Or::make(op->a, op->b));
    }
};

// Assume everything is already in DNF (OR of AND). Does not handle the case
// when OR is inside an AND (AND (A OR B))
class SplitOrs : public HalideNfmConverter {
public:
    vector<vector<Expr>> result;
    SplitOrs() {}
private:
    using HalideNfmConverter::visit;

    void visit(const EQ *op) {
        vector<Expr> ands;
        ands.push_back(EQ::make(op->a, op->b));
        result.push_back(ands);
    }

    void visit(const NE *op) {
        vector<Expr> ands;
        ands.push_back(NE::make(op->a, op->b));
        result.push_back(ands);
    }

    void visit(const LT *op) {
        vector<Expr> ands;
        ands.push_back(LT::make(op->a, op->b));
        result.push_back(ands);
    }

    void visit(const LE *op) {
        vector<Expr> ands;
        ands.push_back(LE::make(op->a, op->b));
        result.push_back(ands);
    }

    void visit(const GT *op) {
        vector<Expr> ands;
        ands.push_back(GT::make(op->a, op->b));
        result.push_back(ands);
    }

    void visit(const GE *op) {
        vector<Expr> ands;
        ands.push_back(GE::make(op->a, op->b));
        result.push_back(ands);
    }

    void visit(const And *op) {
        SplitAnds split;
        split.mutate(And::make(op->a, op->b));
        result.push_back(split.result);
    }

    void visit(const Or *op) {
        mutate(op->a);
        mutate(op->b);
    }
};

// Use distributive rule to convert the expression into sum of multiply, e.g.
// a*(b+c) >= 0 into a*b + a*c >= 0
class DistributeMul : public HalideNfmConverter {
public:
    vector<Expr> result; // sum of mul expr
    DistributeMul() {}
private:
    bool first_entry = true;
    using HalideNfmConverter::visit;

    void visit(const IntImm *op) {
        if (first_entry) {
            assert(result.size() == 0);
            result.push_back(Expr(op->value));
        } else {
            HalideNfmConverter::visit(op);
        }
        first_entry = false;
    }

    void visit(const Variable *op) {
        if (first_entry) {
            assert(result.size() == 0);
            result.push_back(Variable::make(op->type, op->name));
        } else {
            HalideNfmConverter::visit(op);
        }
        first_entry = false;
    }

    void visit(const Add *op) {
        first_entry = false;
        result.clear();
        vector<Expr> a_result;
        vector<Expr> b_result;
        Expr a = mutate(op->a);
        a_result.swap(result);
        Expr b = mutate(op->b);
        b_result.swap(result);

        bool select_a = a_result.size() > 0;
        bool select_b = b_result.size() > 0;
        if (select_a && select_b) {
            for (auto val: a_result) {
                if (!is_zero(val)) {
                    result.push_back(val);
                }
            }
            for (auto val: b_result) {
                if (!is_zero(val)) {
                    result.push_back(val);
                }
            }
        } else if (select_a) {
            for (auto val: a_result) {
                if (!is_zero(val)) {
                    result.push_back(val);
                }
            }
            result.push_back(b);
        } else if (select_b) {
            result.push_back(a);
            for (auto val: b_result) {
                if (!is_zero(val)) {
                    result.push_back(val);
                }
            }
        } else {
            if (!is_zero(a)) {
                result.push_back(a);
            }
            if (!is_zero(b)) {
                result.push_back(b);
            }
        }
        if (result.size() == 1) {
            expr = result[0];
        } else if (result.size() > 1) {
            expr = Add::make(result[0], result[1]);
            for (size_t i = 2; i < result.size(); ++i) {
                expr = Add::make(expr, result[i]);
            }
        } else {
            expr = make_zero(a.type());
        }
    }

    void visit(const Sub *op) {
        first_entry = false;
        result.clear();
        vector<Expr> a_result;
        vector<Expr> b_result;
        Expr a = mutate(op->a);
        a_result.swap(result);
        Expr b = mutate(op->b);
        b_result.swap(result);

        bool select_a = a_result.size() > 0;
        bool select_b = b_result.size() > 0;
        if (select_a && select_b) {
            for (auto val: a_result) {
                if (!is_zero(val)) {
                    result.push_back(val);
                }
            }
            for (auto val: b_result) {
                if (!is_zero(val)) {
                    result.push_back(-1*val);
                }
            }
        } else if (select_a) {
            for (auto val: a_result) {
                if (!is_zero(val)) {
                    result.push_back(val);
                }
            }
            result.push_back(-1*b);
        } else if (select_b) {
            result.push_back(a);
            for (auto val: b_result) {
                if (!is_zero(val)) {
                    result.push_back(-1*val);
                }
            }
        } else {
            if (!is_zero(a)) {
                result.push_back(a);
            }
            if (!is_zero(b)) {
                result.push_back(-1*b);
            }
        }
        if (result.size() == 1) {
            expr = result[0];
        } else if (result.size() > 1) {
            expr = Add::make(result[0], result[1]);
            for (size_t i = 2; i < result.size(); ++i) {
                expr = Add::make(expr, result[i]);
            }
        } else {
            expr = make_zero(a.type());
        }
    }

    void visit(const Mul *op) {
        first_entry = false;

        result.clear();
        vector<Expr> a_result;
        vector<Expr> b_result;
        Expr a = mutate(op->a);
        a_result.swap(result);
        Expr b = mutate(op->b);
        b_result.swap(result);

        if (is_zero(a) || is_zero(b)) {
            expr = make_zero(a.type());
            return;
        }

        bool select_a = a_result.size() > 0;
        bool select_b = b_result.size() > 0;
        if (select_a && select_b) {
            for (auto val_a : a_result) {
                if (!is_zero(val_a)) {
                    for (auto val_b : b_result) {
                        if (!is_zero(val_b)) {
                            result.push_back(val_a*val_b);
                        }
                    }
                }
            }
        } else if (select_a) {
            for (auto val_a : a_result) {
                if (!is_zero(val_a)) {
                    result.push_back(b*val_a);
                }
            }
        } else if (select_b) {
            for (auto val_b : b_result) {
                if (!is_zero(val_b)) {
                    result.push_back(a*val_b);
                }
            }
        } else {
            result.push_back(a*b);
        }
        if (result.size() == 1) {
            expr = result[0];
        } else if (result.size() > 1) {
            expr = Add::make(result[0], result[1]);
            for (size_t i = 2; i < result.size(); ++i) {
                expr = Add::make(expr, result[i]);
            }
        } else{
            expr = make_zero(a.type());
        }
    }

    void visit(const Div *op) {
        internal_error << "DistributeMul: can't handle Div\n";
    }
};

class ConvertToNfmStructs : public HalideNfmConverter {
public:
    struct MulTerm {
        int constant_num;
        int constant_denom;
        vector<int> sym_const_exp_num;
        vector<int> sym_const_exp_denom;
        vector<int> dim_exp_num;
        vector<int> dim_exp_denom;

        MulTerm() : constant_num(1), constant_denom(1) {}
        MulTerm(int sym_size, int dim_size)
            : constant_num(1)
            , constant_denom(1)
            , sym_const_exp_num(sym_size, 0)
            , sym_const_exp_denom(sym_size, 0)
            , dim_exp_num(dim_size, 0)
            , dim_exp_denom(dim_size, 0) {}

        void insert_dim_mul(size_t index) {
            assert(index >= 0 && index < dim_exp_num.size());
            dim_exp_num[index] += 1;
        }

        void insert_dim_div(size_t index) {
            assert(index >= 0 && index < dim_exp_denom.size());
            dim_exp_denom[index] += 1;
        }

        void insert_sym_const_mul(size_t index) {
            assert(index >= 0 && index < sym_const_exp_num.size());
            sym_const_exp_num[index] += 1;
        }

        void insert_sym_const_div(size_t index) {
            assert(index >= 0 && index < sym_const_exp_denom.size());
            sym_const_exp_denom[index] += 1;
        }

        void insert_const_mul(int val) {
            constant_num *= val;
        }

        void insert_const_div(int val) {
            constant_denom *= val;
        }
    };

    ConvertToNfmStructs(Expr e, const vector<string>& sym_const, const vector<string>& dim)
                        : dim_names(dim), sym_const_names(sym_const) {
        in_expr = simplify(e);
        //debug(0) << "ConvertToNfmStructs in_expr: " << in_expr << "\n\n";
        AddNewVar new_var_convert;
        in_expr = new_var_convert.mutate(in_expr);
        expr_substitutions = std::move(new_var_convert.get_expr_substitutions());
        for (auto& str : new_var_convert.get_additional_sym_consts()) {
            insert_sym_const(str);
        }

        //debug(0) << "After AddNewVar: " << in_expr << "\n\n";

        PreProcessor process;
        in_expr = process.mutate(in_expr);
        //debug(0) << "After PreProcessor: " << in_expr << "\n\n";

        for (size_t i = 0; i < dim_names.size(); ++i) {
            dim_to_idx[dim_names[i]] = i;
        }
        for (size_t i = 0; i < sym_const_names.size(); ++i) {
            sym_const_to_idx[sym_const_names[i]] = i;
        }
    }

    NfmUnionDomain convert_to_nfm() {
        //debug(0) << "CONVERTING " << in_expr << "\n\n";
        debug(0) << "CONVERTING START: " << simplify(in_expr) << "\n\n";
        NfmUnionDomain union_dom(sym_const_names, dim_names);
        if (!in_expr.defined() || is_one(in_expr)) { // Undefined expression -> no constraint (universe)
            NfmDomain domain(sym_const_names, dim_names);
            union_dom.add_domain(std::move(domain));
            return union_dom;
        }
        if (in_expr.defined() && is_zero(in_expr)) {
            return union_dom;
        }

        // Convert into (in-)equalities
        ConvertToIneqs convert;
        convert.mutate(in_expr);
        let_substitutions = std::move(convert.get_let_substitutions());
        Expr ineqs_expr = convert.get_result();

        /*{
            SplitOrs split;
            split.mutate(ineqs_expr);
            vector<vector<Expr>> dnf = split.result;

            debug(0) << "DNF " << dnf.size() << "\n";
            for (const auto& ands : dnf) {
                std::cout << "Size: " << ands.size() << "\n";
                for (auto& e : ands) {
                    std::cout << "(" << (e) << ") and ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }*/

        // Convert into DNF
        ConvertToDNF convert_dnf;
        ineqs_expr = convert_dnf.mutate(ineqs_expr);

        //debug(0) << "CONVERTING " << ineqs_expr << "\n\n";
        //debug(0) << "CONVERTING " << simplify(ineqs_expr) << "\n\n";

        // Convert into disjunctive normal form (or of ands)
        SplitOrs split;
        split.mutate(ineqs_expr);
        vector<vector<Expr>> dnf = split.result;
        assert(dnf.size() > 0);

        /*debug(0) << "DNF " << dnf.size() << "\n";
        for (const auto& ands : dnf) {
            for (auto& e : ands) {
                std::cout << "(" << simplify(e) << ") and ";
            }
            std::cout << "\n";
        }*/

        for (const auto& ands : dnf) { // For each disjunctive constraint set
            //debug(0) << "DNF CONSTRAINT: " << "\n";
            NfmDomain domain(sym_const_names, dim_names);
            for (const auto& and_term : ands) { // For each constraint within the set
                //debug(0) << "   AND: " << and_term << "\n";
                const EQ *eq_a = and_term.as<EQ>();
                const GE *ge_a = and_term.as<GE>();
                if (eq_a) { // It's an equality
                    NfmPoly poly = convert_constraint_to_nfm_helper(eq_a->a, true);
                    NfmConstraint cst(poly, true);
                    //debug(0) << "Constraint: " << cst.to_string() << "\n";
                    domain.add_constraint(std::move(cst));
                } else if (ge_a) { // It's an inequality
                    NfmPoly poly = convert_constraint_to_nfm_helper(ge_a->a, false);
                    NfmConstraint cst(poly, false);
                    //debug(0) << "Constraint: " << cst.to_string() << "\n";
                    domain.add_constraint(std::move(cst));
                } else {
                    assert(false); // Shouldn't have had reached here
                }
            }
            //debug(0) << "Domain: \n" << domain.to_string() << "\n\n";
            union_dom.add_domain(std::move(domain));
        }
        union_dom.update_coeff_space(std::move(NfmSpace(sym_const_names)));
        //debug(0) << "\nUnion Domain (" << union_dom.get_domains().size() << "): \n" << union_dom.to_string() << "\n\n";
        //debug(0) << "\nUnion Domain (" << union_dom.get_domains().size() << ")\n";
        /*union_dom.sort();
        for (const auto& dom : union_dom.get_domains()) {
            debug(0) << dom << "\n";
        }
        debug(0) << "\n";*/

        //NfmUnionDomain simplified_isl = union_dom.simplify();
        //debug(0) << "Union Domain (AFTER SIMPLIFY using ISL) (" << simplified_isl.get_domains().size() << "): \n" << simplified_isl.to_string() << "\n";
        /*debug(0) << "Union Domain (AFTER SIMPLIFY using ISL) (" << simplified_isl.get_domains().size() << "): \n";
        simplified_isl.sort();
        for (const auto& dom : simplified_isl.get_domains()) {
            debug(0) << dom << "\n";
        }
        debug(0) << "\n";*/

        NfmUnionDomain simplified_union_dom = NfmSolver::nfm_union_domain_simplify(union_dom);
        //debug(0) << "Union Domain (AFTER SIMPLIFY using NFM) (" << simplified_union_dom.get_domains().size() << "): \n" << simplified_union_dom.to_string() << "\n\n";
        debug(0) << "Union Domain (AFTER SIMPLIFY using NFM) (" << simplified_union_dom.get_domains().size() << "): \n";
        simplified_union_dom.sort();
        for (const auto& dom : simplified_union_dom.get_domains()) {
            debug(0) << dom << "\n";
            debug(0) << "    Context: " << dom.get_context_domain() << "\n";
        }
        debug(0) << "\n\n";
        return simplified_union_dom;
    }

    const map<string, Expr>& get_expr_substitutions() const {
        return expr_substitutions;
    }

    const vector<pair<string, Expr>>& get_let_substitutions() const {
        return let_substitutions;
    }

private:
    MulTerm mul_term;
    bool is_div = false;

    using HalideNfmConverter::visit;

    Expr in_expr; // Need to convert this into a representable form in NFM
    vector<string> dim_names;
    vector<string> sym_const_names;
    map<string, int> dim_to_idx; // Mapping of dim to idx in dim_names
    map<string, int> sym_const_to_idx;  // Mapping of sym const to idx in sym_const_names

    vector<string> additional_sym_consts;
    map<string, Expr> expr_substitutions;
    vector<pair<string, Expr>> let_substitutions;

    string to_string(const MulTerm& term) {
        ostringstream stream;
        if (term.constant_denom == 1) {
            stream << term.constant_num;
        } else {
            stream << "(" << term.constant_num << "/" << term.constant_denom << ")";
        }
        for (size_t i = 0; i < term.sym_const_exp_num.size(); ++i) {
            int exponent = term.sym_const_exp_num[i] - term.sym_const_exp_denom[i];
            if (exponent == 0) {
                continue;
            }
            if (exponent == 1) {
                stream << "*" << sym_const_names[i];
            } else if (exponent != 1) {
                stream << "*(" << sym_const_names[i] << "^" << exponent << ")";
            }
        }
        for (size_t i = 0; i < term.dim_exp_num.size(); ++i) {
            int exponent = term.dim_exp_num[i] - term.dim_exp_denom[i];
            if (exponent == 0) {
                continue;
            }
            if (exponent == 1) {
                stream << "*" << dim_names[i];
            } else if (exponent != 1) {
                stream << "*(" << dim_names[i] << "^" << exponent << ")";
            }
        }
        return stream.str();
    }

    void insert_sym_const(const string& var) {
        sym_const_names.push_back(var);
        size_t idx = sym_const_names.size()-1;
        sym_const_to_idx[var] = idx;
    }

    int get_dim_idx(const string& var) {
        auto it = dim_to_idx.find(var);
        if (it != dim_to_idx.end()) {
            return it->second;
        }
        return -1;
    }

    int get_sym_const_idx(const string& var) {
        auto it = sym_const_to_idx.find(var);
        if (it != sym_const_to_idx.end()) {
            return it->second;
        }
        return -1;
    }

    void reset_mul_term() {
        mul_term = MulTerm(sym_const_names.size(), dim_names.size());
    }

    void insert_mul_term_mul(const string& var) {
        //std::cout << "insert_mul_term_mul: " << var << "\n";
        int dim_idx = get_dim_idx(var);
        if (dim_idx >= 0) {
            //std::cout << "insert_mul_term_mul dim_idx: " << dim_idx << "\n";
            mul_term.insert_dim_mul(dim_idx);
        } else {
            int sym_idx = get_sym_const_idx(var);
            //std::cout << "insert_mul_term_mul sym_idx: " << sym_idx << "\n";
            if (sym_idx < 0) {
                // If you can't find it in sym const either, there is a chance that
                // the simplify function has replaced the var with var.s
                user_assert(ends_with(var, ".s")) << "var: " << var << "\n";
                string var = var.substr(0, var.size()-2);
                dim_idx = get_dim_idx(var);
                //std::cout << "insert_mul_term_mul var: " << var << "; dim_idx: " << dim_idx << "\n";
                if (dim_idx >= 0) {
                    mul_term.insert_dim_mul(dim_idx);
                    return;
                } else {
                    sym_idx = get_sym_const_idx(var);
                }
                if (sym_idx < 0) {
                    // If still can't find it, it's probably temp var from division
                    // normalization. We add it to the end of the sym_vars
                    debug(0) << "ConvertToNfmStructs: adding new symbolic constants " << var << "\n";
                    sym_const_names.push_back(var);
                    sym_idx = sym_const_names.size()-1;
                    sym_const_to_idx[var] = sym_idx;
                    debug(0) << "DONE ConvertToNfmStructs: adding new symbolic constants " << var << "\n";
                }
            }
            user_assert(sym_idx >= 0) << "insert_mul_term_mul var: " << var << "\n";
            //std::cout << "mul_term.insert_sym_const_mul(sym_idx)\n";
            mul_term.insert_sym_const_mul(sym_idx);
        }
    }

    void insert_mul_term_div(const string& var) {
        int dim_idx = get_dim_idx(var);
        if (dim_idx >= 0) {
            mul_term.insert_dim_div(dim_idx);
        } else {
            int sym_idx = get_sym_const_idx(var);
            assert(sym_idx >= 0);
            mul_term.insert_sym_const_div(sym_idx);
        }
    }

    void insert_mul_term_const_mul(int val) {
        mul_term.insert_const_mul(val);
    }

    void insert_mul_term_const_div(int val) {
        mul_term.insert_const_div(val);
    }

    NfmPoly convert_constraint_to_nfm_helper(Expr lhs, bool is_equality) {
        //debug(0) << "convert_constraint_to_nfm_helper: " << lhs << "\n";
        // Convert into summation of multiplication term
        DistributeMul dist;
        dist.mutate(lhs);

        vector<MulTerm> terms;

        vector<int> sym_const_exp_mul(sym_const_names.size(), 0);
        vector<int> dim_exp_mul(dim_names.size(), 0);
        int lcm_val = 1;

        // Determine the lowest common multiplier to normalize the division
        for (const auto& term: dist.result) {
            //debug(0) << "NFM Term: " << term << "\n";
            reset_mul_term();
            mutate(term); // Multiplication terms of each add element
            //debug(0) << "MulTerm: " << to_string(mul_term) << "\n";

            lcm_val = non_neg_lcm(lcm_val, mul_term.constant_denom);
            //debug(0) << "non_neg_lcm: " << lcm_val << "\n";
            vector_max_exp(sym_const_exp_mul, mul_term.sym_const_exp_denom);
            //debug(0) << "vector_max_exp: dim_exp_mul and mul_term.dim_exp_denom\n";
            vector_max_exp(dim_exp_mul, mul_term.dim_exp_denom);
            terms.push_back(std::move(mul_term));
        }

        NfmPoly polynom(sym_const_names, dim_names);
        for (const auto& term : terms) {

            int constant = (lcm_val*term.constant_num)/term.constant_denom;
            vector<int> sym_const_exp =
                term.sym_const_exp_num - term.sym_const_exp_denom + sym_const_exp_mul;
            vector<int> dim_exp =
                term.dim_exp_num - term.dim_exp_denom + dim_exp_mul;

            // There shouldn't be any division factor left after normalization
            assert((lcm_val*term.constant_num) % term.constant_denom == 0);
            assert(std::all_of(sym_const_exp.begin(), sym_const_exp.end(),
                [](int i) { return i>=0; }));
            assert(std::all_of(dim_exp.begin(), dim_exp.end(),
                [](int i) { return i>=0; }));

            polynom = polynom.add(constant, sym_const_exp, NFM_UNKNOWN, dim_exp);
        }
        return polynom;
    }

    void visit(const IntImm *op) {
        //debug(0) << "IntImm: " << op->value << "\n";
        if (!is_div) {
            insert_mul_term_const_mul(op->value);
        } else {
            insert_mul_term_const_div(op->value);
        }
    }

    void visit(const Variable *op) {
        //debug(0) << "Variable: " << op->name << "\n";
        if (!is_div) {
            insert_mul_term_mul(op->name);
        } else {
            insert_mul_term_div(op->name);
        }
    }

    void visit(const Mul *op) {
        //debug(0) << "Mul: (" << op->a << ")*(" << op->b << ")\n";
        mutate(op->a);
        mutate(op->b);
    }

    void visit(const Add *op) {
        //debug(0) << "Add: (" << op->a << ")+(" << op->b << ")\n";
        HalideNfmConverter::visit(op);
    }

    void visit(const Sub *op) {
        //debug(0) << "Sub: (" << op->a << ")-(" << op->b << ")\n";
        HalideNfmConverter::visit(op);
    }

    // TODO: Expand this to handle division by expr as well; need to split into
    // 2 ORs (one for when the divisor is positive and one for negative case)
    // TODO: can't handle this case: 2/(M+N)
    void visit(const Div *op) {
        internal_error << "ConvertToNfmStructs: can't handle Div\n";
        /*assert(op->b.as<IntImm>() != NULL)
            << "can only handle division by integer\n";*/
        //debug(0) << "Div: (" << op->a << ")/(" << op->b << ")\n";
        /*is_div = false;
        mutate(op->a);
        is_div = true;
        mutate(op->b);
        is_div = false;*/
    }
};

}

void ir_nfm_test() {
    Expr x = Variable::make(Int(32), "x");
    Expr y = Variable::make(Int(32), "y");
    Expr z = Variable::make(Int(32), "z");
    Expr w = Variable::make(Int(32), "w");
    Expr s = Variable::make(Int(32), "s");
    Expr M = Variable::make(Int(32), "M");
    Expr N = Variable::make(Int(32), "N");

    Expr sub = 2*x - y*(4-z+w%2) - select(x>=0, y , z);
    Expr eq_expr = EQ::make(x + 2*y, z);
    Expr gt_expr = GT::make(4*x + y, 10);
    Expr lt_expr = GT::make(z, 13);
    Expr let_expr = Let::make("y", 2*x - 12, gt_expr);
    Expr select_expr = Select::make(2*z - 1 >= 2, 3*x, 4*x);
    Expr select_expr1 = Select::make(x >= 2, select_expr, -12*x+y);
    Expr select_expr2 = Select::make(z >= 0, x, y);
    Expr eq_select = EQ::make(x, select_expr);
    Expr min_expr = EQ::make(z, Min::make(3*x-1, 4*x-y));
    Expr min_expr1 = LT::make(z, 2*w+Min::make(3*x-1, Min::make(y, 4*x-y)));
    Expr or_expr = Or::make(4*x>=2, And::make(3*x+2>=1, z>=3));
    Expr gt_select = GT::make(y, select_expr);
    Expr lt_select1 = LT::make(select_expr2, select_expr1);
    Expr eq_select_select = EQ::make(w, select_expr1);
    Expr sub_select_select = 2*x - y*(4-z+w%2) - select(x>=0, y , z) - select_expr1;
    Expr lt_select_select = LT::make(w, select_expr1);
    Expr max_expr = GT::make(z, Max::make(3*x-1, y+w));
    Expr max_expr1 = LT::make(z, 2*w+Max::make(3*x-1, Min::make(y, y+z)));
    Expr div_expr = LT::make(M, Div::make(Min::make(x,y), Min::make(z,w)));

    ConvertToIneqs convert;
    //Expr expr = Select::make(M >= 0, select(N >= 0, x, y), z);
    /*Expr expr = EQ::make(w, (((((((((y - x) + 16)/16)*(((z - s) + 16)/16)) + -1)/(((y - x) + 16)/16))*16) + s) + 15));
    convert.mutate(expr);
    std::cout << convert.get_result() << "\n\n";*/

    Expr expr = w <= max(((0 + select((x < 22), (x + 1), x)) - 1), y);
    NumOpsCounter counter;
    counter.mutate(expr);
    std::cout << "Expr: " << expr << "\n";
    std::cout << "Count: " << counter.get_count() << "\n";
    //Expr expr = max(min((((max((max(s, 1) + -1), 0)*16) + y) + 15), x), z);
    /*std::cout << "Before mutating: " << expr << "\n\n";
    PreProcessor process;
    expr = process.mutate(expr);
    std::cout << "After mutating: " << expr << "\n\n";
    std::cout << "After mutating (SIMPLIFY): " << simplify(expr) << "\n\n";*/

    /*std::cout << convert.mutate(x) << "\n\n";
    std::cout << convert.mutate(eq_expr) << "\n\n";
    std::cout << convert.mutate(gt_expr) << "\n\n";
    std::cout << convert.mutate(lt_expr) << "\n\n";
    std::cout << convert.mutate(eq_select) << "\n\n";
    std::cout << convert.mutate(let_expr) << "\n\n";
    std::cout << convert.mutate(or_expr) << "\n\n";
    std::cout << convert.mutate(select_expr) << "\n\n";
    std::cout << convert.mutate(gt_select) << "\n\n";
    std::cout << convert.mutate(lt_select1) << "\n\n";
    std::cout << convert.mutate(select_expr1) << "\n\n";
    std::cout << convert.mutate(eq_select_select) << "\n\n";
    std::cout << convert.mutate(sub) << "\n\n";
    std::cout << convert.mutate(sub_select_select) << "\n\n";
    std::cout << convert.mutate(lt_select_select) << "\n\n";
    std::cout << convert.mutate(min_expr) << "\n\n";
    std::cout << convert.mutate(min_expr1) << "\n\n";
    std::cout << convert.mutate(max_expr) << "\n\n";*/
    //std::cout << convert.mutate(max_expr1) << "\n\n";

    /*std::cout << "Converting: " << div_expr << "\n\n";
    Expr expr = convert.mutate(div_expr);
    std::cout << "After mutating: \n" << expr << "\n\n";

    std::cout << "Converting into DNF\n";
    SplitOrs split_or;
    split_or.mutate(expr);
    vector<vector<Expr>> result = split_or.result;
    for (auto ands : result) {
        for (auto e : ands) {
          std::cout << "  " << e << "\n";
        }
        std::cout << "OR\n";
    }*/

    /*Expr ands_expr = And::make(z>2, And::make(And::make(y>2, x>3), And::make(y-z>0, x-w>=1)));
    //Expr ands_expr = z > 2;
    SplitAnds split_ands;
    split_ands.mutate(ands_expr);
    vector<Expr> result = split_ands.result;
    for (auto e : result) {
        std::cout << e << "\n";ConvertToNfmStructsConvertToNfmStructs
    }*/

    //Expr expr = Or::make(And::make(z>2, z<1), Or::make(And::make(y>2, x>3), And::make(y-z>0, x-w>=1)));
    //Expr expr = z > 2;
    /*Expr expr = convert.mutate(max_expr1);
    SplitOrs split_or;
    split_or.mutate(expr);
    vector<vector<Expr>> result = split_or.result;
    for (auto ands : result) {
        for (auto e : ands) {
          std::cout << e << "\n";
        }
        std::cout << "\n";
    }*/

    //vector<string> sym_const_names = {"dummy"};
    //vector<string> dim_names = {"x", "y", "z", "w"};
    //Expr test_expr = LT::make(z, 2*w+Max::make(3*x-1, Min::make(y, y+z)));
    //Expr test_expr = GE::make(w, Max::make(x, y));
    //Expr test_expr = EQ::make(w, Mul::make(Div::make(x-y, 4), 2*z));
    //Expr test_expr = EQ::make(w, min(min(min(x, y), z), z));
    //Expr test_expr = EQ::make(2*w, min(z, -z));
    //Expr test_expr = EQ::make(2*w, min(z, x)-min(4*x, 3*y));
    //Expr test_expr = EQ::make(2*w, min(min(4*x, 3*y), 6*y+7*z));
    //Expr test_expr = LT::make(M, Div::make(Min::make(x,y), Min::make(z,w)));
    /*Expr test_expr = Let::make("N", cast(Int(32), ceil(1000.000000f/cast(Float(32), M))), EQ::make(w, x+y));

    std::cout << "TESTING: " << test_expr << "\n";
    CollectVars collect({"x", "y", "z", "w"});
    collect.mutate(test_expr);
    for (const auto& name : collect.get_dims()) {
        std::cout << "DIM: " << name << "\n";
    }
    for (auto& name : collect.get_sym_consts()) {
        std::cout << "SYM CONSTANT: " << name << "\n";
    }*/
    /*NfmUnionDomain union_dom = convert_halide_expr_to_nfm_union_domain(
        test_expr, collect.get_sym_consts(), collect.get_dims());
    //std::cout << "\n\nNfmUnionDomain: " << union_dom << "\n";
    std::cout << "\n\nNfmUnionDomain: \n";
    for (const auto& dom : union_dom.get_domains()) {
        std::cout << dom << "\n";
    }
    //Expr res = convert_nfm_union_domain_to_halide_expr(Int(32), union_dom);
    //  std::cout << "\n\nConvert back to Expr: " << res << "\n\n";
    //std::cout << simplify(res);
    Interval interval = convert_nfm_union_domain_to_halide_interval(Int(32), union_dom, "w");
    std::cout << "\nInterval:\n  Min: " << interval.min << "\n  Max: " << interval.max << "\n";*/
}

vector<vector<Expr>> split_expr_into_dnf(Expr expr) {
    SplitOrs split;
    split.mutate(expr);
    return split.result;
}

Expr convert_interval_to_expr(const Interval& interval) {
    //std::cout << "CONVERTING INTERVAL TO EXPR: min: " << interval.min << "; max: " << interval.max << "\n";
    Expr expr;
    if (interval.min.defined() && interval.max.defined()) {
        Expr var = Variable::make(interval.min.type(), interval.var);
        expr = And::make(LE::make(interval.min, var), LE::make(var, interval.max));
    } else if (interval.min.defined()) {
        Expr var = Variable::make(interval.min.type(), interval.var);
        expr = LE::make(interval.min, var);
    } else if (interval.max.defined()) {
        Expr var = Variable::make(interval.max.type(), interval.var);
        expr = LE::make(var, interval.max);
    }
    //std::cout << "  expr: " << expr << "\n";
    return expr;
}

Expr convert_interval_to_expr_lower_bound(const Interval& interval) {
    Expr expr;
    if (interval.min.defined()) {
        Expr var = Variable::make(interval.min.type(), interval.var);
        expr = LE::make(interval.min, var);
    }
    return expr;
}

Expr convert_interval_to_expr_upper_bound(const Interval& interval) {
    Expr expr;
    if (interval.max.defined()) {
        Expr var = Variable::make(interval.max.type(), interval.var);
        expr = LE::make(var, interval.max);
    }
    return expr;
}

NfmUnionDomain convert_halide_interval_to_nfm_union_domain(
        const Interval& interval,
        const vector<Dim>& loop_dims,
        map<string, Expr> *expr_substitutions,
        vector<pair<string, Expr>> *let_substitutions) {
    Expr expr = convert_interval_to_expr(interval);
    return convert_halide_expr_to_nfm_union_domain(
        expr, loop_dims, expr_substitutions, let_substitutions);
}

NfmUnionDomain convert_halide_interval_to_nfm_union_domain(
        const Interval& interval,
        const vector<string>& sym_const,
        const vector<string>& dim,
        map<string, Expr> *expr_substitutions,
        vector<pair<string, Expr>> *let_substitutions) {
    Expr expr = convert_interval_to_expr(interval);
    return convert_halide_expr_to_nfm_union_domain(
        expr, sym_const, dim, expr_substitutions, let_substitutions);
}


NfmUnionDomain convert_halide_expr_to_nfm_union_domain(
        const Expr& expr,
        const vector<Dim>& loop_dims, // Loop var from innermost to outermost
        map<string, Expr> *expr_substitutions,
        vector<pair<string, Expr>> *let_substitutions) {
    CollectVars collect(loop_dims);
    collect.mutate(expr);
    vector<string> dim = collect.get_dims();
    vector<string> sym_const = collect.get_sym_consts();
    return convert_halide_expr_to_nfm_union_domain(
        expr, sym_const, dim, expr_substitutions, let_substitutions);
}

NfmUnionDomain convert_halide_expr_to_nfm_union_domain(
        const Expr& expr,
        const vector<string>& sym_const,
        const vector<string>& dim, // Loop var from outermost to innermost
        map<string, Expr> *expr_substitutions,
        vector<pair<string, Expr>> *let_substitutions) {
    Expr simplified_expr = simplify(expr);
    //Expr simplified_expr = expr;
    ConvertToNfmStructs convert(simplified_expr, sym_const, dim);
    NfmUnionDomain union_dom = convert.convert_to_nfm();
    if (expr_substitutions != NULL) {
        for (auto& iter : convert.get_expr_substitutions()) {
            (*expr_substitutions).emplace(iter.first, iter.second);
        }
    }
    if (let_substitutions != NULL) {
        for (auto& iter : convert.get_let_substitutions()) {
            (*let_substitutions).push_back(std::make_pair(iter.first, iter.second));
        }
    }
    return union_dom;
}


}
}
