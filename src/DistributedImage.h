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
vector<int> get_buffer_bounds(Func f, vector<Expr> &allocated_extents_parameterized, vector<Expr> &allocated_mins_parameterized,
                              vector<Expr> &local_extents_parameterized, vector<Expr> &local_mins_parameterized,
                              vector<int> &global_mins, vector<int> &local_mins, vector<int> &local_extents);

}

template<typename T>
class DistributedImage {
    // Full extents of the global image.
    vector<int> full_extents;
    // The actual buffer allocated may be of a larger size than what
    // is needed, due to boundary conditions, so we must keep track of
    // what was allocated separately from the region used.
    vector<int> allocated_extents, local_extents;
    // The mins of this buffer in global coordinates. Note that this
    // is the min of the allocated region, not the min of the region
    // used.
    vector<int> global_mins;
    // The mins of this buffer in local coordinates. Similar to
    // extents, the allocated mins may be different from the local
    // mins. In particular, the allocated min is always 0, but the
    // local min may be >= 0.
    vector<int> local_mins;
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
        // Determine size of the buffer to be allocated, and also
        // mins/extents of this buffer in parameterized global
        // coordinates (i.e. parameterized by rank and number of
        // processors).
        vector<Expr> allocated_extents_parameterized, allocated_mins_parameterized,
            local_extents_parameterized, local_mins_parameterized;
        allocated_extents =
            Internal::get_buffer_bounds(wrapper, allocated_extents_parameterized, allocated_mins_parameterized,
                                        local_extents_parameterized, local_mins_parameterized,
                                        global_mins, local_mins, local_extents);
        Buffer b(type_of<T>(), full_extents, NULL, param.name());
        b.set_distributed(allocated_extents, allocated_extents_parameterized, allocated_mins_parameterized,
                          local_extents_parameterized, local_mins_parameterized);
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

    int allocated_min(int dim) const {
        return global_mins[dim];
    }
    
    int global_extent(int dim) const {
        internal_assert(!full_extents.empty());
        internal_assert(dim < full_extents.size());
        return full_extents[dim];
    }
    int global_width() const { return global_extent(0); }
    int global_height() const { return global_extent(1); }
    int global_channels() const { return global_extent(2); }

    int extent(int dim) const {
        internal_assert(!local_extents.empty());
        internal_assert(dim < local_extents.size());
        return local_extents[dim];
    }
    int width() const { return extent(0); }
    int height() const { return extent(1); }
    int channels() const { return extent(2); }

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
        internal_assert(!global_mins.empty());
        internal_assert(dim < local_mins.size());
        internal_assert(local_mins[dim] >= 0);
        return global_mins[dim] + local_mins[dim] + c;
    }

    /** Return the local coordinate of dimension 'dim' corresponding
     * to the global coordinate value c. */
    int local(int dim, int c) const {
        internal_assert(!global_mins.empty());
        internal_assert(dim < local_mins.size());
        internal_assert(local_mins[dim] >= 0);
        return c - (global_mins[dim] + local_mins[dim]);
    }

    /** Return true if the global x coordinate resides on this
     * rank. */
    bool mine(int x) const {
        internal_assert(!global_mins.empty() && !local_extents.empty());
        const int lx = local(0, x);
        return lx >= 0 && lx < extent(0);
    }

    bool mine(int x, int y) const {
        internal_assert(!global_mins.empty() && !local_extents.empty());
        internal_assert(global_mins.size() == 2);
        const int lx = local(0, x), ly = local(1, y);
        return lx >= 0 && lx < extent(0) && ly >= 0 && ly < extent(1);
    }

    bool mine(int x, int y, int z) const {
        internal_assert(!global_mins.empty() && !local_extents.empty());
        internal_assert(global_mins.size() == 3);
        const int lx = local(0, x), ly = local(1, y), lz = local(2, z);
        return lx >= 0 && lx < extent(0) && ly >= 0 && ly < extent(1) && lz >= 0 && lz < extent(2);
    }

    /** Get a pointer to the element at the min location. */
    NO_INLINE T *data() const {
        return (T *)image.get().host_ptr();
    }

    /** Assuming this image is one-dimensional, get the value of the
     * element at position x */
    const T &operator()(int x) const {
        // We're given a coordinate in local space; convert to allocated space.
        const int ax = x + local_mins[0];
        return image(ax);
    }

    /** Assuming this image is two-dimensional, get the value of the
     * element at position (x, y) */
    const T &operator()(int x, int y) const {
        const int ax = x + local_mins[0], ay = y + local_mins[1];
        return image(ax, ay);
    }

    /** Assuming this image is three-dimensional, get the value of the
     * element at position (x, y, z) */
    const T &operator()(int x, int y, int z) const {
        const int ax = x + local_mins[0], ay = y + local_mins[1], az = z + local_mins[2];
        return image(ax, ay, az);
    }

    /** Assuming this image is four-dimensional, get the value of the
     * element at position (x, y, z, w) */
    const T &operator()(int x, int y, int z, int w) const {
        const int ax = x + local_mins[0], ay = y + local_mins[1],
            az = z + local_mins[2], aw = w + local_mins[3];
        return image(ax, ay, az, aw);
    }

    /** Assuming this image is one-dimensional, get a reference to the
     * element at position x */
    T &operator()(int x) {
        // We're given a coordinate in local space; convert to allocated space.
        const int ax = x + local_mins[0];
        return image(ax);
    }

    /** Assuming this image is two-dimensional, get the value of the
     * element at position (x, y) */
    T &operator()(int x, int y) {
        const int ax = x + local_mins[0], ay = y + local_mins[1];
        return image(ax, ay);
    }

    /** Assuming this image is three-dimensional, get the value of the
     * element at position (x, y, z) */
    T &operator()(int x, int y, int z) {
        const int ax = x + local_mins[0], ay = y + local_mins[1], az = z + local_mins[2];
        return image(ax, ay, az);
    }

    /** Assuming this image is four-dimensional, get the value of the
     * element at position (x, y, z, w) */
    T &operator()(int x, int y, int z, int w) {
        const int ax = x + local_mins[0], ay = y + local_mins[1],
            az = z + local_mins[2], aw = w + local_mins[3];
        return image(ax, ay, az, aw);
    }

    Expr operator()(Expr x) const {
        // Note we do not convert these to allocated space: these
        // expressions need access to the full allocated space for
        // e.g. ghost zone access.
        return param(x);
    }

    Expr operator()(Expr x, Expr y) const {
        return param(x, y);
    }

    Expr operator()(Expr x, Expr y, Expr z) const {
        return param(x, y, z);
    }

    Expr operator()(Expr x, Expr y, Expr z, Expr w) const {
        return param(x, y, z, w);
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

    operator Buffer() const {
        return (Buffer)image;
    }
};
}

#endif
