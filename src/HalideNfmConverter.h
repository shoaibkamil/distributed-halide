#ifndef HALIDE_HALIDE_NFM_CONVERTER_H
#define HALIDE_HALIDE_NFM_CONVERTER_H

#include <iostream>
#include <sstream>
#include <set>

#include <nfm_constraint.h>
#include <nfm_domain.h>
#include <nfm_polynom.h>

#include "Bounds.h"
#include "IRMutator.h"
#include "IRVisitor.h"
#include "Module.h"
#include "Schedule.h"
#include "Util.h"

namespace Halide {
namespace Internal {

class NumOpsCounter : public IRMutator {
public:
    NumOpsCounter() : count(0) {}
    int get_count() { return count; }
protected:
    using IRMutator::visit;

    void visit(const IntImm *) { count = 0; }
    void visit(const UIntImm *) { count = 0; }
    void visit(const FloatImm *) { count = 0; }
    void visit(const Variable *) { count = 0; }

    void visit(const Cast *op) { update_simple(op->value, COST_CAST); }
    void visit(const Add *op) { update_simple(op->a, op->b, COST_ADD); }
    void visit(const Sub *op) { update_simple(op->a, op->b, COST_SUB); }
    void visit(const Mul *op) { update_simple(op->a, op->b, COST_MUL); }
    void visit(const Div *op) { update_simple(op->a, op->b, COST_DIV); }
    void visit(const Mod *op) { update_simple(op->a, op->b, COST_MOD); }

    void visit(const Min *op) { update_min_max(op->a, op->b); }
    void visit(const Max *op) { update_min_max(op->a, op->b); }

    void visit(const EQ *op) { update_simple(op->a, op->b, COST_COMPARE); }
    void visit(const NE *op) { update_simple(op->a, op->b, COST_COMPARE); }
    void visit(const LT *op) { update_simple(op->a, op->b, COST_COMPARE); }
    void visit(const LE *op) { update_simple(op->a, op->b, COST_COMPARE); }
    void visit(const GT *op) { update_simple(op->a, op->b, COST_COMPARE); }
    void visit(const GE *op) { update_simple(op->a, op->b, COST_COMPARE); }
    void visit(const And *op) { update_simple(op->a, op->b, COST_AND); }
    void visit(const Or *op) { update_simple(op->a, op->b, COST_OR); }
    void visit(const Not *op) { update_simple(op->a, COST_NOT); }

    void visit(const Select *op) {
        int cond = 0;
        mutate(op->condition);
        swap(cond, count);

        int left = 0;
        mutate(op->true_value);
        swap(left, count);

        int right = 0;
        mutate(op->false_value);
        swap(right, count);

        count = std::max(left, right) + cond + COST_RETURN;
    }

    void visit(const Call *op) {
        for (const auto& expr : op->args) {
            update_simple(expr, COST_CALL);
        }
    }

    void visit(const Let *op) {
        update_simple(op->value, 0);
        update_simple(op->body, 0);
    }

    void visit(const StringImm *) { error("StringImm"); }
    void visit(const Load *) { error("Load"); }
    void visit(const Ramp *) { error("Ramp"); }
    void visit(const Broadcast *) { error("Broadcast"); }
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
    static const int COST_CAST = 1;
    static const int COST_ADD = 1;
    static const int COST_SUB = 1;
    static const int COST_MUL = 1;
    static const int COST_DIV = 1;
    static const int COST_MOD = 1;
    static const int COST_RETURN = 1;
    static const int COST_COMPARE = 1;
    static const int COST_AND = 1;
    static const int COST_OR = 1;
    static const int COST_NOT = 1;
    static const int COST_CALL = 1;

    int count;

    void error(const std::string& op_name) {
        internal_error << "NumOpsCounter can't handle " << op_name << "\n";
    }

    void update_simple(const Expr& a, int cost) {
        int val = 0;
        mutate(a);
        swap(val, count);

        count = val + cost;
    }

    void update_simple(const Expr& a, const Expr&b, int cost) {
        int left = 0;
        mutate(a);
        swap(left, count);

        int right = 0;
        mutate(b);
        swap(right, count);

        count = left + right + cost;
    }

    void update_min_max(const Expr& a, const Expr&b) {
        int left = 0;
        mutate(a);
        swap(left, count);

        int right = 0;
        mutate(b);
        swap(right, count);

        count = left + right + COST_COMPARE + COST_RETURN;
    }

    void swap(int& a, int&b) {
        int temp = a;
        a = b;
        b = temp;
    }
};

class CollectVars : public IRMutator {
public:
    CollectVars(const std::vector<Dim>& loop_dims) {
        // Dim from schedule is in reverse order (innermost to outermost dimension)
        for (int i = loop_dims.size()-1; i >= 0; --i) {
            const auto& dim = loop_dims[i];
            dim_names_.push_back(dim.var);
        }
    }
    CollectVars(const std::vector<std::string>& dim_names) { // Loop var from outermost to innermost
        for (const auto& name : dim_names) {
            dim_names_.push_back(name);
        }
    }
    CollectVars() {};

    std::vector<std::string> get_dims() {
        if (dims_.size() > 0) {
            return dims_;
        }
        dims_ = std::vector<std::string>(dims_temp_.begin(), dims_temp_.end());
        std::sort(dims_.begin(), dims_.end(),
            [this] (const std::string& dim1, const std::string& dim2) {
                return compare_dims(dim1, dim2);
        });

        if (dims_.size() == 0) {
            dims_.push_back("_dummy_dim");
        }
        return dims_;
    }

    std::vector<std::string> get_sym_consts() {
        if (sym_consts_.size() == 0) {
            sym_consts_.push_back("_dummy_sym");
        }
        return sym_consts_;
    }

    std::vector<std::string> get_let_assignments() {
        return let_assignments_;
    }

private:
    using IRMutator::visit;
    // Everything that ends with any of the dim_names_ is considered as
    // loop variable; otherwise it's a symbolic constant. Exception: XXX.base is
    // a loop variable (occurs when we have the split case). XXX is one of the
    // loop var in dim_names_.
    // If dim_names_ is empty, treat everything as dims since we can't have
    // empty dimension
    std::vector<std::string> dim_names_;

    std::vector<std::string> dims_;
    std::vector<std::string> sym_consts_;

    std::vector<std::string> let_assignments_;

    // Temporary storage
    std::set<std::string> dims_temp_;
    std::map<std::string, int> sym_consts_temp_;
    std::map<std::string, int> let_assignments_temp_;

    int find_dim_index(const std::string& dim) const {
        std::string var = dim;
        /*TODO: handle the special case (vector)
        if (ends_with(dim, ".base")) {
            var = dim.substr(0, dim.size()-5);
        }*/
        for (size_t i = 0; i < dim_names_.size(); ++i) {
            if (ends_with(var, dim_names_[i])) {
                return (int)i;
            }
        }
        return -1;
    }

    bool compare_dims(const std::string& dim1, const std::string& dim2) {
        int idx_dim1 = find_dim_index(dim1);
        int idx_dim2 = find_dim_index(dim2);
        return idx_dim1 < idx_dim2;
    }

    bool is_dim(const std::string& name) const {
        if (dim_names_.empty()) {
            return true;
        }
        std::string var = name;
        /*if (ends_with(name, ".base")) {
            var = name.substr(0, name.size()-5);
        }*/
        for (const auto& suffix : dim_names_) {
            if (ends_with(var, suffix)) {
                return true;
            }
        }
        return false;
    }

    void insert(const std::string& name) {
        if (is_dim(name)) {
            dims_temp_.insert(name);
        } else {
            if (sym_consts_temp_[name] == 0) {
                sym_consts_.push_back(name);
                sym_consts_temp_[name] += 1;
            }
        }
    }

    void visit(const Let *op) {
        // op->name should come after whatever vars in value, e.g. in let M = N + 10,
        // the order should be {N, M} instead of {M, N} since M is dependent on N
        //debug(0) << "CollectVars: let(" << op->name << " = " << op->value << "); " << op->body << "\n";
        if (let_assignments_temp_[op->name] == 0) {
            let_assignments_.push_back(op->name);
            let_assignments_temp_[op->name] += 1;
        }
        Expr value = mutate(op->value);
        insert(op->name);
        Expr body = mutate(op->body);
        if (value.same_as(op->value) &&
            body.same_as(op->body)) {
            expr = op;
        } else {
            expr = Let::make(op->name, value, body);
        }
    }

    void visit(const Variable *op) {
        expr = op;
        insert(op->name);
    }
};

void ir_nfm_test();

std::vector<std::vector<Expr>> split_expr_into_dnf(Expr expr);

Expr convert_interval_to_expr(const Interval& interval);

Expr convert_interval_to_expr_lower_bound(const Interval& interval);

Expr convert_interval_to_expr_upper_bound(const Interval& interval);

Nfm::Internal::NfmUnionDomain convert_halide_interval_to_nfm_union_domain(
    const Interval& interval,
    const std::vector<Dim>& loop_dims,
    std::map<std::string, Expr> *expr_substitutions=NULL,
    std::vector<std::pair<std::string, Expr>> *let_substitutions=NULL);

Nfm::Internal::NfmUnionDomain convert_halide_interval_to_nfm_union_domain(
    const Interval& interval,
    const std::vector<std::string>& sym_const,
    const std::vector<std::string>& dim,
    std::map<std::string, Expr> *expr_substitutions=NULL,
    std::vector<std::pair<std::string, Expr>> *let_substitutions=NULL);


Nfm::Internal::NfmUnionDomain convert_halide_expr_to_nfm_union_domain(
    const Expr& expr,
    const std::vector<Dim>& loop_dims,
    std::map<std::string, Expr> *expr_substitutions=NULL,
    std::vector<std::pair<std::string, Expr>> *let_substitutions=NULL);

Nfm::Internal::NfmUnionDomain convert_halide_expr_to_nfm_union_domain(
    const Expr& expr,
    const std::vector<std::string>& sym_const,
    const std::vector<std::string>& dim,
    std::map<std::string, Expr> *expr_substitutions=NULL,
    std::vector<std::pair<std::string, Expr>> *let_substitutions=NULL);

}
}

#endif
