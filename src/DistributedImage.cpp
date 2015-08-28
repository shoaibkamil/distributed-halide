#include "DistributedImage.h"
#include "IRMutator.h"
#include "Substitute.h"

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

vector<int> get_buffer_bounds(const string &name, Func f, const vector<int> &output_extents, const vector<int> &full_extents,
                              vector<Expr> &symbolic_extents, vector<Expr> &symbolic_mins,
                              vector<int> &mins, vector<int> &capped_local_extents) {
    vector<int> bounds;
    Stmt s = partial_lower(f);
    GetBoxes get;
    s.accept(&get);

    int rank = 0, num_processors = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    const Box &b = get.boxes.at(name);
    map<string, Expr> env;
    for (int i = 0; i < (int)b.size(); i++) {
        Expr sz = b[i].max - b[i].min + 1;
        symbolic_extents.push_back(sz);
        symbolic_mins.push_back(b[i].min);
        int output_extent = output_extents.empty() ? full_extents[i] : output_extents[i];
        env[f.name() + ".extent." + std::to_string(i)] = output_extent;
        env[f.name() + ".min." + std::to_string(i)] = 0;
        sz = simplify(Let::make("Rank", rank, Let::make("NumProcessors", num_processors, substitute(env, sz))));

        const int *dim = as_const_int(sz);
        internal_assert(dim != NULL) << sz;
        bounds.push_back(*dim);
        const int *min = as_const_int(simplify(Let::make("Rank", rank, Let::make("NumProcessors", num_processors, substitute(env, b[i].min)))));
        internal_assert(min != NULL);
        mins.push_back(*min);
    }

    s = partial_lower(f, true);
    GetBoxes get_capped;
    s.accept(&get_capped);
    const Box &b_capped = get_capped.boxes.at(name);
    for (int i = 0; i < (int)b_capped.size(); i++) {
        Expr sz = b_capped[i].max - b_capped[i].min + 1;
        sz = simplify(Let::make("Rank", rank, Let::make("NumProcessors", num_processors, substitute(env, sz))));
        const int *dim = as_const_int(sz);
        internal_assert(dim != NULL) << sz;
        capped_local_extents.push_back(*dim);
    }

    return bounds;
}
}
}
