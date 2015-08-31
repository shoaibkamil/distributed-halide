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
        IRVisitor::visit(op);
        map<string, Box> r = boxes_required(op);
        for (auto it : r) {
            Box b;
            // TODO: this can be quite slow with large
            // environments. The goal is to get the Box into terms of
            // the outermost variables (the global output buffer
            // variables).
            for (unsigned i = 0; i < it.second.size(); i++) {
                Expr min = it.second[i].min, max = it.second[i].max;
                for (auto let = lets.rbegin(); let != lets.rend(); ++let) {
                    min = simplify(substitute(let->first, let->second, min));
                    max = simplify(substitute(let->first, let->second, max));
                }
                b.push_back(Interval(min, max));
            }
            boxes[it.first] = b;
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

}
}
