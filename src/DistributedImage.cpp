#include "DistributedImage.h"

using std::string;
using std::map;
using std::vector;

namespace Halide {
namespace Internal {

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
    s = bounds_inference(s, outputs, order, env, func_bounds);
    s = allocation_bounds_inference(s, env, func_bounds);
    s = uniquify_variable_names(s);
    s = storage_folding(s);
    s = simplify(s, false);
    s = distribute_loops_only(s);
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
        internal_assert(dim != NULL);
        bounds.push_back(*dim);
        mins.push_back(simplify(Let::make("Rank", rank, Let::make("SliceSize", slice_size, b[i].min))));
    }
    return bounds;
}
}
}
