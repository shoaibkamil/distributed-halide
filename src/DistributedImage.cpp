#include "DistributedImage.h"
#include "IRMutator.h"

using std::string;
using std::map;
using std::vector;

namespace Halide {
namespace Internal {

namespace {

class GetBoxes : public IRVisitor {
    vector<std::pair<string, Expr> > lets;
    Scope<Interval> loop_var_bounds;

    Box simplify_box(const Box &b) {
        Box result;
        // TODO: this can be quite slow with large
        // environments. The goal is to get the Box into terms of
        // the outermost variables (the global output buffer
        // variables).
        for (unsigned i = 0; i < b.size(); i++) {
            Expr min = b[i].min, max = b[i].max;
            for (auto let = lets.rbegin(); let != lets.rend(); ++let) {
                min = simplify(substitute(let->first, let->second, min));
                max = simplify(substitute(let->first, let->second, max));
            }
            result.push_back(Interval(min, max));
        }
        return result;
    }

    void get_bounds(Stmt s) {
        map<string, Box> required = boxes_required(s, loop_var_bounds);
        for (auto it : required) {
            const auto bounds = boxes.find(it.first);
            if (bounds == boxes.end()) {
                boxes[it.first] = simplify_box(it.second);
            } else {
                Box existing = bounds->second;
                merge_boxes(existing, it.second);
                boxes[it.first] = simplify_box(existing);
             }
        }
    }
public:
    using IRVisitor::visit;

    void visit(const LetStmt *let) {
        lets.push_back(std::make_pair(let->name, let->value));
        IRVisitor::visit(let);
        lets.pop_back();
    }

    void visit(const Let *let) {
        lets.push_back(std::make_pair(let->name, let->value));
        IRVisitor::visit(let);
        lets.pop_back();
    }

    virtual void visit(const For *op) {
        loop_var_bounds.push(op->name, Interval(op->min, op->min + op->extent - 1));
        get_bounds(op);
        // Don't need to recurse on the body because the required
        // region will be redundant for nested accesses.
        loop_var_bounds.pop(op->name);
    }

    map<string, Box> boxes;
};

// Constant-folds the pow_32 extern call.
class ConstantFoldPow : public IRMutator {
    float to_float(Expr e) const {
        float result = 0;
        if (const double *v = as_const_float(e)) {
            result = (float)*v;
        } else if (const int64_t *v = as_const_int(e)) {
            result = (float)*v;
        } else if (const uint64_t *v = as_const_uint(e)) {
            result = (float)*v;
        } else {
            internal_assert(false) << e;
        }
        return result;
    }

    Expr compute_pow32(Expr a, Expr b) const {
        float fa = to_float(a), fb = to_float(b);
        double result = std::pow(fa, fb);
        return FloatImm::make(Float(32), (float)result);
    }
public:
    using IRMutator::visit;

    void visit(const Call *op) {
        if (op->call_type == Call::Extern && op->name == "pow_f32") {
            internal_assert(op->args.size() == 2);
            Expr a = mutate(op->args[0]), b = mutate(op->args[1]);
            expr = compute_pow32(a, b);
        } else {
            IRMutator::visit(op);
        }
    }
};

}

// Lower the given function enough to get bounds information on
// input buffers with respect to rank and number of MPI
// processors.
Stmt partial_lower(Func f, bool cap_extents) {
    Target t = get_target_from_environment();
    map<string, Function> env;
    vector<Function> outputs(1, f.function());
    for (Function f : outputs) {
        map<string, Function> more_funcs = find_transitive_calls(f);
        env.insert(more_funcs.begin(), more_funcs.end());
    }
    vector<string> order = realization_order(outputs, env);
    bool any_memoized = false;
    Stmt s = schedule_functions(outputs, order, env, any_memoized, !t.has_feature(Target::NoAsserts));
    FuncValueBounds func_bounds = compute_function_value_bounds(order, env);
    s = distribute_loops(s, env, func_bounds, cap_extents);
    s = bounds_inference(s, outputs, order, env, func_bounds);
    s = allocation_bounds_inference(s, env, func_bounds);
    s = uniquify_variable_names(s);
    return s;
}

map<string, Box> get_boxes(Func f, bool cap_extents) {
    Stmt s = partial_lower(f, cap_extents);
    GetBoxes get;
    s.accept(&get);
    return get.boxes;
}

// Nested distribution code makes use of the pow_f32 extern call,
// which is not constant folded by the normal simplification
// process. We do it manually here.
Expr fold_pow(Expr e) {
    return ConstantFoldPow().mutate(e);
}

}
}
