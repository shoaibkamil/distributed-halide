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
    int local_min[4], local_extent[4], local_stride[4];
    Expr symbolic_min[4], symbolic_extent[4], symbolic_stride[4];

    void set_distributed(int x_size_local, int y_size_local,
                         int z_size_local, int w_size_local,
                         Expr x_size_symbolic, Expr y_size_symbolic,
                         Expr z_size_symbolic, Expr w_size_symbolic) {
        uint64_t size = 1;
        if (x_size_local) size *= x_size_local;
        if (y_size_local) size *= y_size_local;
        if (z_size_local) size *= z_size_local;
        if (w_size_local) size *= w_size_local;
        size *= buf.elem_size;

        local_extent[0] = x_size_local;
        local_extent[1] = y_size_local;
        local_extent[2] = z_size_local;
        local_extent[3] = w_size_local;
        local_stride[0] = 1;
        local_stride[1] = x_size_local;
        local_stride[2] = x_size_local*y_size_local;
        local_stride[3] = x_size_local*y_size_local*z_size_local;
        local_min[0] = 0;
        local_min[1] = 0;
        local_min[2] = 0;
        local_min[3] = 0;

        symbolic_extent[0] = x_size_symbolic;
        symbolic_extent[1] = y_size_symbolic;
        symbolic_extent[2] = z_size_symbolic;
        symbolic_extent[3] = w_size_symbolic;
        symbolic_stride[0] = 1;
        symbolic_stride[1] = x_size_symbolic;
        symbolic_stride[2] = x_size_symbolic*y_size_symbolic;
        symbolic_stride[3] = x_size_symbolic*y_size_symbolic*z_size_symbolic;
        symbolic_min[0] = 0;
        symbolic_min[1] = 0;
        symbolic_min[2] = 0;
        symbolic_min[3] = 0;

        distributed = true;

        size = size + 32;
        allocation = (uint8_t *)realloc(allocation, (size_t)size);
        user_assert(allocation) << "Out of memory allocating buffer " << name << " of size " << size << "\n";
        buf.host = allocation;
        while ((size_t)(buf.host) & 0x1f) buf.host++;
    }

    BufferContents(Type t, int x_size, int y_size, int z_size, int w_size,
                   uint8_t* data, const std::string &n) :
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
            size = size + 32;
            check_buffer_size(size, name);
            allocation = (uint8_t *)calloc(1, (size_t)size);
            user_assert(allocation) << "Out of memory allocating buffer " << name << " of size " << size << "\n";
            buf.host = allocation;
            while ((size_t)(buf.host) & 0x1f) buf.host++;
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

Expr size_or_zero(const std::vector<Expr> &sizes, size_t index) {
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
               uint8_t* data, const std::string &name) :
    contents(new Internal::BufferContents(t, x_size, y_size, z_size, w_size, data,
                                          make_buffer_name(name, this))) {
}

Buffer::Buffer(Type t, const std::vector<int32_t> &sizes,
               uint8_t* data, const std::string &name) :
    contents(new Internal::BufferContents(t,
                                          size_or_zero(sizes, 0),
                                          size_or_zero(sizes, 1),
                                          size_or_zero(sizes, 2),
                                          size_or_zero(sizes, 3),
                                          data,
                                          make_buffer_name(name, this))) {
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

void Buffer::set_distributed(const std::vector<int> &local_sizes,
                             const std::vector<Expr> &symbolic_extents) {
    user_assert(defined()) << "Buffer is undefined\n";
    contents.ptr->set_distributed(size_or_zero(local_sizes, 0),
                                  size_or_zero(local_sizes, 1),
                                  size_or_zero(local_sizes, 2),
                                  size_or_zero(local_sizes, 3),
                                  size_or_zero(symbolic_extents, 0),
                                  size_or_zero(symbolic_extents, 1),
                                  size_or_zero(symbolic_extents, 2),
                                  size_or_zero(symbolic_extents, 3));
}

Expr Buffer::local_extent(int dim) const {
    user_assert(defined()) << "Buffer is undefined\n";
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    user_assert(contents.ptr->distributed) << "Calling local function on non-distributed buffer.";
    return contents.ptr->symbolic_extent[dim];
}

Expr Buffer::local_stride(int dim) const {
    user_assert(defined()) << "Buffer is undefined\n";
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    user_assert(contents.ptr->distributed) << "Calling local function on non-distributed buffer.";
    return contents.ptr->symbolic_stride[dim];
}

Expr Buffer::local_min(int dim) const {
    user_assert(defined()) << "Buffer is undefined\n";
    user_assert(dim >= 0 && dim < 4) << "We only support 4-dimensional buffers for now";
    user_assert(contents.ptr->distributed) << "Calling local function on non-distributed buffer.";
    return contents.ptr->symbolic_min[dim];
}

}
