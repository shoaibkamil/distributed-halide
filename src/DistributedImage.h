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
#include "UniquifyVariableNames.h"
#include "Util.h"
#include <map>
#include <sstream>

using std::string;
using std::map;
using std::vector;

namespace Halide {

namespace Internal {

// Lower the given function enough to get bounds information on
// input buffers with respect to rank and number of MPI
// processors.
Stmt partial_lower(Func f, bool cap_extents=false);
vector<int> get_buffer_bounds(Func f, const vector<int> &full_extents,
                              vector<Expr> &symbolic_extents, vector<Expr> &symbolic_mins,
                              vector<int> &mins, vector<int> &capped_local_extents);

}

template<typename T>
class DistributedImage {
    vector<int> full_extents;
    vector<int> local_extents, capped_local_extents;
    vector<Expr> symbolic_extents, symbolic_mins;
    vector<int> mins;
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
        if (name.empty()) {
            param = ImageParam(type_of<T>(), full_extents.size());
        } else {
            param = ImageParam(type_of<T>(), full_extents.size(), name);
        }
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
        local_extents = Internal::get_buffer_bounds(wrapper, full_extents, symbolic_extents, symbolic_mins, mins,
                                                    capped_local_extents);
        Buffer b(type_of<T>(), full_extents, NULL, param.name());
        b.set_distributed(local_extents, symbolic_extents, symbolic_mins);
        param.set(b);
        image = Image<T>(b);

        // Remove the explicit bounds from the wrapper function.
        wrapper.function().schedule().bounds().clear();
    }

    /** Return the underlying buffer. Only relevant for jitting. */
    Buffer get_buffer() {
        return param.get();
    }

    int dimensions() const { return image.dimensions(); }
    int global_extent(int dim) const { return image.extent(dim); }
    int global_width() const { return image.width(); }
    int global_height() const { return image.height(); }
    int global_channels() const { return image.channels(); }

    int extent(int dim) const { return capped_local_extents[dim]; }
    int width() const { return capped_local_extents[0]; }
    int height() const { return capped_local_extents[1]; }
    int channels() const { return capped_local_extents[2]; }

    /** Return the global x coordinate corresponding to the local x
     * coordinate. */
    int global(int x) const {
        return global(0, x);
    }

    /** Return the local x coordinate corresponding to the global x
     * coordinate. */
    int local(int x) const {
        return local(0, x);
    }

    /** Return the global coordinate of dimension 'dim' corresponding
     * to the local coordinate value c. */
    int global(int dim, int c) const {
        internal_assert(!mins.empty());
        return mins[dim] + c;
    }

    /** Return the local coordinate of dimension 'dim' corresponding
     * to the global coordinate value c. */
    int local(int dim, int c) const {
        internal_assert(!mins.empty());
        return c - mins[dim];
    }

    /** Return true if the global x coordinate resides on this
     * rank. */
    bool mine(int x) const {
        internal_assert(!mins.empty() && !local_extents.empty());
        return x >= mins[0] && x < (mins[0] + local_extents[0]);
    }

    bool mine(int x, int y) const {
        internal_assert(!mins.empty() && !local_extents.empty());
        internal_assert(mins.size() == 2);
        bool myx = x >= mins[0] && x < (mins[0] + local_extents[0]);
        bool myy = y >= mins[1] && y < (mins[1] + local_extents[1]);
        return myx && myy;
    }

    bool mine(int x, int y, int z) const {
        internal_assert(!mins.empty() && !local_extents.empty());
        internal_assert(mins.size() == 3);
        bool myx = x >= mins[0] && x < (mins[0] + local_extents[0]);
        bool myy = y >= mins[1] && y < (mins[1] + local_extents[1]);
        bool myz = z >= mins[2] && z < (mins[2] + local_extents[2]);
        return myx && myy && myz;
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

    /** Get the name of this image. */
    const std::string &name() {
        return image.name();
    }
};
}

#endif
