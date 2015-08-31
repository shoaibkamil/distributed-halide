#include "DistributedImage.h"
#include "IRMutator.h"

using std::string;
using std::map;
using std::vector;

namespace Halide {
namespace Internal {

namespace {

class ReplaceVariables : public IRMutator {
    const Scope<Expr> &replacements;
public:
    ReplaceVariables(const Scope<Expr> &r) : replacements(r) {}

    using IRMutator::visit;
    void visit(const Variable *op) {
        IRMutator::visit(op);
        if (replacements.contains(op->name)) {
            expr = mutate(replacements.get(op->name));
        }
    }
};

// Simplify the given box, using the given environment of variable
// to value.
Box simplify_box(const Box &b, const Scope<Expr> &env) {
    Box result(b.size());
    ReplaceVariables replace(env);
    for (unsigned i = 0; i < b.size(); i++) {
        Expr min = replace.mutate(b[i].min),
            max = replace.mutate(b[i].max);
        result[i] = Interval(simplify(min), simplify(max));
    }
    return result;
}

class GetBoxes : public IRVisitor {
public:
    Scope<Expr> env;
    using IRVisitor::visit;

    void visit(const LetStmt *let) {
        env.push(let->name, let->value);
        IRVisitor::visit(let);
        env.pop(let->name);
    }

    void visit(const Let *let) {
        env.push(let->name, let->value);
        IRVisitor::visit(let);
        env.pop(let->name);
    }

    virtual void visit(const For *op) {
        IRVisitor::visit(op);
        map<string, Box> r = boxes_required(op);
        for (auto it : r) {
            boxes[it.first] = simplify_box(it.second, env);
        }
    }

    map<string, Box> boxes;
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
    Stmt s = schedule_functions(outputs, order, env, !t.has_feature(Target::NoAsserts));
    FuncValueBounds func_bounds = compute_function_value_bounds(order, env);
    s = distribute_loops_only(s, env, cap_extents);
    s = bounds_inference(s, outputs, order, env, func_bounds);
    return s;
}

map<string, Box> get_boxes(Func f, bool cap_extents) {
    Stmt s = partial_lower(f, cap_extents);
    GetBoxes get;
    s.accept(&get);
    return get.boxes;
}

}
}
