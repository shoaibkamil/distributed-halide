#include "Buffer.h"
#include "Debug.h"
#include "Error.h"
#include "IROperator.h"
#include "JITModule.h"
#include "runtime/HalideRuntime.h"

namespace Halide {
namespace Internal {

namespace {
void check_buffer_size(uint64_t bytes, const std::string &name) {
    user_assert(bytes < (1UL << 31)) << "Total size of buffer " << name << " exceeds 2^31 - 1\n";
}
}


struct BufferContents {
    /** The buffer_t object we're wrapping. */
    buffer_t buf;

    /** The type of the allocation. buffer_t's don't currently track this so we do it here. */
    Type type;

    /** If we made the allocation ourselves via a Buffer constructor,
     * and hence should delete it when this buffer dies, then this
     * pointer is set to the memory we need to free. Otherwise it's
     * NULL. */
    uint8_t *allocation;

    /** How many Buffer objects point to this BufferContents */
    mutable RefCount ref_count;

    /** What is the name of the buffer? Useful for debugging symbols. */
    std::string name;

    bool distributed;
    int allocated_min[4], allocated_extent[4];
    Expr symbolic_allocated_min[4], symbolic_allocated_extent[4];
    Expr symbolic_local_min[4], symbolic_local_extent[4];

    void set_distributed(int x_size_allocated, int y_size_allocated,
                         int z_size_allocated, int w_size_allocated,
                         Expr x_size_allocated_symbolic, Expr y_size_allocated_symbolic,
                         Expr z_size_allocated_symbolic, Expr w_size_allocated_symbolic,
                         Expr x_min_allocated_symbolic, Expr y_min_allocated_symbolic,
                         Expr z_min_allocated_symbolic, Expr w_min_allocated_symbolic,
                         Expr x_size_local_symbolic, Expr y_size_local_symbolic,
                         Expr z_size_local_symbolic, Expr w_size_local_symbolic,
                         Expr x_min_local_symbolic, Expr y_min_local_symbolic,
                         Expr z_min_local_symbolic, Expr w_min_local_symbolic) {

        uint64_t size = 1;
        if (x_size_allocated) size *= x_size_allocated;
        if (y_size_allocated) size *= y_size_allocated;
        if (z_size_allocated) size *= z_size_allocated;
        if (w_size_allocated) size *= w_size_allocated;
        size *= buf.elem_size;

        allocated_min[0] = 0;
        allocated_min[1] = 0;
        allocated_min[2] = 0;
        allocated_min[3] = 0;
        allocated_extent[0] = x_size_allocated;
        allocated_extent[1] = y_size_allocated;
        allocated_extent[2] = z_size_allocated;
        allocated_extent[3] = w_size_allocated;

        // TODO: This is a bit of a hack, because the buffer extent
        // values are global, but now we are making the stride values
        // local. The proper way to do this would be to make
        // StorageFlattening aware of distributed buffers so that it
        // can use local strides instead of global strides. For now,
        // this does the trick.
        buf.stride[0] = 1;
        buf.stride[1] = x_size_allocated;
        buf.stride[2] = x_size_allocated*y_size_allocated;
        buf.stride[3] = x_size_allocated*y_size_allocated*z_size_allocated;

        symbolic_allocated_extent[0] = x_size_allocated_symbolic;
        symbolic_allocated_extent[1] = y_size_allocated_symbolic;
        symbolic_allocated_extent[2] = z_size_allocated_symbolic;
        symbolic_allocated_extent[3] = w_size_allocated_symbolic;
        symbolic_allocated_min[0] = x_min_allocated_symbolic;
        symbolic_allocated_min[1] = y_min_allocated_symbolic;
        symbolic_allocated_min[2] = z_min_allocated_symbolic;
        symbolic_allocated_min[3] = w_min_allocated_symbolic;

        symbolic_local_extent[0] = x_size_local_symbolic;
        symbolic_local_extent[1] = y_size_local_symbolic;
        symbolic_local_extent[2] = z_size_local_symbolic;
        symbolic_local_extent[3] = w_size_local_symbolic;
        symbolic_local_min[0] = x_min_local_symbolic;
        symbolic_local_min[1] = y_min_local_symbolic;
        symbolic_local_min[2] = z_min_local_symbolic;
        symbolic_local_min[3] = w_min_local_symbolic;

        distributed = true;

        size = size + 32;
        allocation = (uint8_t *)realloc(allocation, (size_t)size);
        user_assert(allocation) << "Out of memory allocating buffer " << name << " of size " << size << "\n";
        buf.host = allocation;
        while ((size_t)(buf.host) & 0x1f) buf.host++;
    }

    BufferContents(Type t, int x_size, int y_size, int z_size, int w_size,
                   uint8_t* data, const std::string &n, bool alloc) :
        type(t), allocation(NULL), name(n.empty() ? unique_name('b') : n) {
        user_assert(t.width == 1) << "Can't create of a buffer of a vector type";
        buf.elem_size = t.bytes();
        uint64_t size = 1;
        if (x_size) {
            size *= x_size;
            check_buffer_size(size, name);
        }
        if (y_size) {
            size *= y_size;
            check_buffer_size(size, name);
        }
        if (z_size) {
            size *= z_size;
            check_buffer_size(size, name);
        }
        if (w_size) {
            size *= w_size;
            check_buffer_size(size, name);
        }
        size *= buf.elem_size;
        check_buffer_size(size, name);

        if (!data) {
            if (alloc) {
                size = size + 32;
                check_buffer_size(size, name);
                allocation = (uint8_t *)calloc(1, (size_t)size);
                user_assert(allocation) << "Out of memory allocating buffer " << name << " of size " << size << "\n";
                buf.host = allocation;
                while ((size_t)(buf.host) & 0x1f) buf.host++;
            }
        } else {
            buf.host = data;
        }
        buf.dev = 0;
        buf.host_dirty = false;
        buf.dev_dirty = false;
        buf.extent[0] = x_size;
        buf.extent[1] = y_size;
        buf.extent[2] = z_size;
        buf.extent[3] = w_size;
        buf.stride[0] = 1;
        buf.stride[1] = x_size;
        buf.stride[2] = x_size*y_size;
        buf.stride[3] = x_size*y_size*z_size;
        buf.min[0] = 0;
        buf.min[1] = 0;
        buf.min[2] = 0;
        buf.min[3] = 0;

        distributed = false;
    }

    BufferContents(Type t, const buffer_t *b, const std::string &n) :
        type(t), allocation(NULL), name(n.empty() ? unique_name('b') : n) {
        buf = *b;
        user_assert(t.width == 1) << "Can't create of a buffer of a vector type";
        distributed = false;
    }
};

template<>
EXPORT RefCount &ref_count<BufferContents>(const BufferContents *p) {
    return p->ref_count;
}

template<>
EXPORT void destroy<BufferContents>(const BufferContents *p) {
    int error = halide_device_free(NULL, const_cast<buffer_t *>(&p->buf));
    user_assert(!error) << "Failed to free device buffer\n";
    free(p->allocation);

    delete p;
}

}

namespace {
int32_t size_or_zero(const std::vector<int32_t> &sizes, size_t index) {
    return (index < sizes.size()) ? sizes[index] : 0;
}

Expr expr_or_zero(const std::vector<Expr> &sizes, size_t index) {
    return (index < sizes.size()) ? sizes[index] : 0;
}

std::string make_buffer_name(const std::string &n, Buffer *b) {
    if (n.empty()) {
        return Internal::make_entity_name(b, "Halide::Buffer", 'b');
    } else {
        return n;
    }
}
}

Buffer::Buffer(Type t, int x_size, int y_size, int z_size, int w_size,
               uint8_t* data, const std::string &name, bool alloc) :
    contents(new Internal::BufferContents(t, x_size, y_size, z_size, w_size, data,
                                          make_buffer_name(name, this), alloc)) {
}

Buffer::Buffer(Type t, const std::vector<int32_t> &sizes,
               uint8_t* data, const std::string &name, bool alloc) :
    contents(new Internal::BufferContents(t,
                                          size_or_zero(sizes, 0),
                                          size_or_zero(sizes, 1),
                                          size_or_zero(sizes, 2),
                                          size_or_zero(sizes, 3),
                                          data,
                                          make_buffer_name(name, this),
                                          alloc)) {
    user_assert(sizes.size() <= 4) << "Buffer dimensions greater than 4 are not supported.";
}

Buffer::Buffer(Type t, const buffer_t *buf, const std::string &name) :
    contents(new Internal::BufferContents(t, buf,
                                          make_buffer_name(name, this))) {
}

void *Buffer::host_ptr() const {
    user_assert(defined()) << "Buffer is undefined\n";
    return (void *)contents.ptr->buf.host;
}

buffer_t *Buffer::raw_buffer() const {
    user_assert(defined()) << "Buffer is undefined\n";
    return &(contents.ptr->buf);
}

uint64_t Buffer::device_handle() const {
    user_assert(defined()) << "Buffer is undefined\n";
    return contents.ptr->buf.dev;
}

bool Buffer::host_dirty() const {
    user_assert(defined()) << "Buffer is undefined\n";
    return contents.ptr->buf.host_dirty;
}

void Buffer::set_host_dirty(bool dirty) {
    user_assert(defined()) << "Buffer is undefined\n";
    contents.ptr->buf.host_dirty = dirty;
}

bool Buffer::device_dirty() const {
    user_assert(defined()) << "Buffer is undefined\n";
    return contents.ptr->buf.dev_dirty;
}

void Buffer::set_device_dirty(bool dirty) {
    user_assert(defined()) << "Buffer is undefined\n";
    contents.ptr->buf.dev_dirty = dirty;
}

int Buffer::dimensions() const {
    for (int i = 0; i < 4; i++) {
        if (extent(i) == 0) return i;
    }
    return 4;
}

int Buffer::extent(int dim) const {
    user_assert(defined()) << "Buffer is undefined\n";
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    return contents.ptr->buf.extent[dim];
}

int Buffer::stride(int dim) const {
    user_assert(defined());
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    return contents.ptr->buf.stride[dim];
}

int Buffer::min(int dim) const {
    user_assert(defined()) << "Buffer is undefined\n";
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    return contents.ptr->buf.min[dim];
}

void Buffer::set_min(int m0, int m1, int m2, int m3) {
    user_assert(defined()) << "Buffer is undefined\n";
    contents.ptr->buf.min[0] = m0;
    contents.ptr->buf.min[1] = m1;
    contents.ptr->buf.min[2] = m2;
    contents.ptr->buf.min[3] = m3;
}

Type Buffer::type() const {
    user_assert(defined()) << "Buffer is undefined\n";
    return contents.ptr->type;
}

bool Buffer::same_as(const Buffer &other) const {
    return contents.same_as(other.contents);
}

bool Buffer::defined() const {
    return contents.defined();
}

const std::string &Buffer::name() const {
    return contents.ptr->name;
}

Buffer::operator Argument() const {
    return Argument(name(), Argument::InputBuffer, type(), dimensions());
}

int Buffer::copy_to_host() {
    return halide_copy_to_host(NULL, raw_buffer());
}

int Buffer::copy_to_device() {
  return halide_copy_to_device(NULL, raw_buffer(), NULL);
}

int Buffer::free_dev_buffer() {
    return halide_device_free(NULL, raw_buffer());
}

bool Buffer::distributed() const {
    user_assert(defined()) << "Buffer is undefined\n";
    return contents.ptr->distributed;
}

void Buffer::set_distributed(const std::vector<int> &allocated_sizes,
                             const std::vector<Expr> &symbolic_allocated_extents,
                             const std::vector<Expr> &symbolic_allocated_mins,
                             const std::vector<Expr> &symbolic_local_extents,
                             const std::vector<Expr> &symbolic_local_mins) {
    user_assert(defined()) << "Buffer is undefined\n";
    contents.ptr->set_distributed(size_or_zero(allocated_sizes, 0),
                                  size_or_zero(allocated_sizes, 1),
                                  size_or_zero(allocated_sizes, 2),
                                  size_or_zero(allocated_sizes, 3),
                                  expr_or_zero(symbolic_allocated_extents, 0),
                                  expr_or_zero(symbolic_allocated_extents, 1),
                                  expr_or_zero(symbolic_allocated_extents, 2),
                                  expr_or_zero(symbolic_allocated_extents, 3),
                                  expr_or_zero(symbolic_allocated_mins, 0),
                                  expr_or_zero(symbolic_allocated_mins, 1),
                                  expr_or_zero(symbolic_allocated_mins, 2),
                                  expr_or_zero(symbolic_allocated_mins, 3),
                                  expr_or_zero(symbolic_local_extents, 0),
                                  expr_or_zero(symbolic_local_extents, 1),
                                  expr_or_zero(symbolic_local_extents, 2),
                                  expr_or_zero(symbolic_local_extents, 3),
                                  expr_or_zero(symbolic_local_mins, 0),
                                  expr_or_zero(symbolic_local_mins, 1),
                                  expr_or_zero(symbolic_local_mins, 2),
                                  expr_or_zero(symbolic_local_mins, 3));
}

Expr Buffer::allocated_extent(int dim) const {
    user_assert(defined()) << "Buffer is undefined\n";
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    user_assert(contents.ptr->distributed) << "Calling local function on non-distributed buffer.";
    return contents.ptr->symbolic_allocated_extent[dim];
}

Expr Buffer::allocated_min(int dim) const {
    user_assert(defined()) << "Buffer is undefined\n";
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    user_assert(contents.ptr->distributed) << "Calling local function on non-distributed buffer.";
    return contents.ptr->symbolic_allocated_min[dim];
}

Expr Buffer::local_extent(int dim) const {
    user_assert(defined()) << "Buffer is undefined\n";
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    user_assert(contents.ptr->distributed) << "Calling local function on non-distributed buffer.";
    return contents.ptr->symbolic_local_extent[dim];
}

Expr Buffer::local_min(int dim) const {
    user_assert(defined()) << "Buffer is undefined\n";
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    user_assert(contents.ptr->distributed) << "Calling local function on non-distributed buffer.";
    return contents.ptr->symbolic_local_min[dim];
}

}
