#include "DistributedImage.h"
#include "IRMutator.h"

using std::string;
using std::map;
using std::vector;

namespace Halide {
namespace Internal {

namespace {

class ReplaceVariable : public IRMutator {
    string name;
    Expr value;
public:
    ReplaceVariable(const string &n, Expr v) :
        name(n), value(v) {}

    using IRMutator::visit;
    void visit(const Variable *op) {
        IRMutator::visit(op);
        if (op->name == name) {
            expr = value;
        }
    }
};

// Simplify the given box, using the given environment of variable
// to value.
Box simplify_box(const Box &b, const Scope<Expr> &env) {
    Box result(b.size());
    for (unsigned i = 0; i < b.size(); i++) {
        Expr min = b[i].min, max = b[i].max;
        for (auto it = env.cbegin(), ite = env.cend(); it != ite; ++it) {
            ReplaceVariable replace(it.name(), it.value());
            min = replace.mutate(min);
            max = replace.mutate(max);
        }
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

Stmt partial_lower(Func f) {
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
    s = distribute_loops_only(s);
    s = bounds_inference(s, outputs, order, env, func_bounds);
    return s;
}

vector<int> get_buffer_bounds(Func f, const vector<int> &full_extents,
                              vector<Expr> &symbolic_extents, vector<Expr> &symbolic_mins,
                              vector<Expr> &mins) {
    vector<int> bounds;
    Stmt s = partial_lower(f);
    GetBoxes get;
    s.accept(&get);
    internal_assert(get.boxes.size() == 1);

    int rank = 0, num_processors = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    const Box &b = get.boxes.begin()->second;
    for (int i = 0; i < (int)b.size(); i++) {
        Expr slice_size = cast(Int(32), ceil(cast(Float(32), full_extents[i]) / num_processors));
        Expr sz = b[i].max - b[i].min + 1;
        symbolic_extents.push_back(sz);
        symbolic_mins.push_back(b[i].min);
        sz = simplify(Let::make("Rank", rank, Let::make("SliceSize", slice_size, sz)));
        const int *dim = as_const_int(sz);
        internal_assert(dim != NULL) << sz;
        bounds.push_back(*dim);
        mins.push_back(simplify(Let::make("Rank", rank, Let::make("SliceSize", slice_size, b[i].min))));
    }
    return bounds;
}
}
}
