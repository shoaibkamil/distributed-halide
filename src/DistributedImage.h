#ifndef HALIDE_DISTRIBUTED_IMAGE_H
#define HALIDE_DISTRIBUTED_IMAGE_H

#include <mpi.h>

#include "ScheduleFunctions.h"
#include "AllocationBoundsInference.h"
#include "BoundsInference.h"
#include "RealizationOrder.h"
#include "StorageFolding.h"
#include "DistributeLoops.h"
#include "Bounds.h"
#include "FindCalls.h"
#include "Func.h"
#include "Image.h"
#include "IR.h"
#include "IRVisitor.h"
#include "Lower.h"
#include "Simplify.h"
#include "Schedule.h"
#include <map>
#include <sstream>

using std::string;
using std::map;
using std::vector;

namespace Halide {

namespace Internal {

class GetBoxes : public IRVisitor {
public:
    using IRVisitor::visit;
    virtual void visit(const For *op) {
        map<string, Box> b = boxes_required(op);
        boxes.insert(b.begin(), b.end());
        IRVisitor::visit(op);
    }

    map<string, Box> boxes;
};

// Lower the given function enough to get bounds information on
// input buffers with respect to rank and number of MPI
// processors.
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
    s = storage_folding(s);
    s = simplify(s, false);
    s = distribute_loops_only(s);
    return s;
}

vector<int> get_buffer_bounds(Func f, const vector<int> &full_extents,
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
        sz = simplify(Let::make("Rank", rank, Let::make("SliceSize", slice_size, sz)));
        const int *dim = as_const_int(sz);
        internal_assert(dim != NULL);
        bounds.push_back(*dim);
        mins.push_back(simplify(Let::make("Rank", rank, Let::make("SliceSize", slice_size, b[i].min))));
    }
    return bounds;
}
}

template<typename T>
class DistributedImage {
    vector<int> full_extents;
    vector<int> local_extents;
    vector<Expr> mins;
    ImageParam param;
    Image<T> image;
    Func wrapper;

public:
    /** Construct an undefined image handle */
    DistributedImage() {}

    /** Define a distributed image with the given dimensions. */
    // @{
    DistributedImage(int x, int y = 0, int z = 0, int w = 0, const std::string &name = "") {
        if (x) full_extents.push_back(x);
        if (y) full_extents.push_back(y);
        if (z) full_extents.push_back(z);
        if (w) full_extents.push_back(w);
        param = ImageParam(type_of<T>(), full_extents.size(), name);
    }
    DistributedImage(int x, const std::string &name) :
        DistributedImage(x, 0, 0, 0, name) {}
    DistributedImage(int x, int y, const std::string &name) :
        DistributedImage(x, y, 0, 0, name) {}
    DistributedImage(int x, int y, int z, const std::string &name) :
        DistributedImage(x, y, z, 0, name) {}
    // @}

    /** Set the domain variables of the image, used for scheduling placement. */
    // @{
    void set_domain(const vector<Var> &vars) {
        if (!wrapper.defined()) {
            wrapper = Func("accessor_" + param.name());
            internal_assert(vars.size() == full_extents.size());
            wrapper(vars) = param(vars);
            for (int i = 0; i < full_extents.size(); i++) {
                wrapper.bound(vars[i], 0, full_extents[i]);
            }
            wrapper.compute_root();
        }
    }

    void set_domain(Var x, Var y, Var z, Var w) {
        vector<Var> vars({x, y, z, w});
        set_domain(vars);
    }

    void set_domain(Var x, Var y, Var z) {
        vector<Var> vars({x, y, z});
        set_domain(vars);
    }

    void set_domain(Var x, Var y) {
        vector<Var> vars({x, y});
        set_domain(vars);
    }

    void set_domain(Var x) {
        vector<Var> vars({x});
        set_domain(vars);
    }
    // @}

    Func &placement() {
        return wrapper;
    }

    /** Allocate memory for the Image portion residing on this rank
     * (determined by the scheduling of placement). Only relevant for
     * jitting. */
    void allocate() {
        internal_assert(!image.defined());
        local_extents = Internal::get_buffer_bounds(wrapper, full_extents, mins);
        Buffer b = Buffer(type_of<T>(), local_extents, NULL, param.name());
        param.set(b);
        image = Image<T>(b);
    }

    /** Return the underlying buffer. Only relevant for jitting. */
    Buffer get_buffer() {
        return param.get();
    }

    int dimensions() const { return image.dimensions(); }
    int extent(int dim) const { return image.extent(dim); }
    int width() const { return image.width(); }
    int height() const { return image.height(); }
    int channels() const { return image.channels(); }

    /** Return the global x coordinate corresponding to the local x
     * coordinate. */
    int global(int x) const {
        return global(0, x);
    }

    /** Return the global coordinate of dimension 'dim' corresponding
     * to the local coordinate value c. */
    int global(int dim, int c) const {
        Expr g = simplify(mins[dim] + c);
        const int *result = as_const_int(g);
        internal_assert(result != NULL);
        return *result;
    }
    
    /** Get a pointer to the element at the min location. */
    NO_INLINE T *data() const {
        return (T *)image.get().host_ptr();
    }

    /** Assuming this image is one-dimensional, get the value of the
     * element at position x */
    const T &operator()(int x) const {
        return image(x);
    }

    /** Assuming this image is two-dimensional, get the value of the
     * element at position (x, y) */
    const T &operator()(int x, int y) const {
        return image(x, y);
    }

    /** Assuming this image is three-dimensional, get the value of the
     * element at position (x, y, z) */
    const T &operator()(int x, int y, int z) const {
        return image(x, y, z);
    }

    /** Assuming this image is four-dimensional, get the value of the
     * element at position (x, y, z, w) */
    const T &operator()(int x, int y, int z, int w) const {
        return image(x, y, z, w);
    }

    /** Assuming this image is one-dimensional, get a reference to the
     * element at position x */
    T &operator()(int x) {
        return image(x);
    }

    /** Assuming this image is two-dimensional, get a reference to the
     * element at position (x, y) */
    T &operator()(int x, int y) {
        return image(x, y);
    }

    /** Assuming this image is three-dimensional, get a reference to the
     * element at position (x, y, z) */
    T &operator()(int x, int y, int z) {
        return image(x, y, z);
    }

    /** Assuming this image is four-dimensional, get a reference to the
     * element at position (x, y, z, w) */
    T &operator()(int x, int y, int z, int w) {
        return image(x, y, z, w);
    }

    Expr operator()(Expr x) const {
        return image(x);
    }

    Expr operator()(Expr x, Expr y) const {
        return image(x, y);
    }

    Expr operator()(Expr x, Expr y, Expr z) const {
        return image(x, y, z);
    }

    Expr operator()(Expr x, Expr y, Expr z, Expr w) const {
        return image(x, y, z, w);
    }

    /** Convert this image to an argument to a halide pipeline. */
    operator Argument() const {
        return (Argument)image;
    }

    /** Convert this image to an argument to an extern stage. */
    operator ExternFuncArgument() const {
        return (ExternFuncArgument)image;
    }

    /** Treating the image as an Expr is equivalent to call it with no
     * arguments. For example, you can say:
     *
     \code
     Image im(10, 10);
     Func f;
     f = im*2;
     \endcode
     *
     * This will define f as a two-dimensional function with value at
     * position (x, y) equal to twice the value of the image at the
     * same location.
     */
    operator Expr() const {
        return (Expr)image;
    }
};
}

#endif
