#include <algorithm>
#include <set>
#include <sstream>

#include "Bounds.h"
#include "Parameter.h"
#include "DistributeLoops.h"
#include "IRMutator.h"
#include "Scope.h"
#include "IRPrinter.h"
#include "Deinterleave.h"
#include "Substitute.h"
#include "IROperator.h"
#include "IREquality.h"
#include "ExprUsesVar.h"
#include "Simplify.h"
#include "Var.h"
#include "Image.h"
// Includes for distribute_loops_test:
#include "DistributedImage.h"
#include "Func.h"
#include "FindCalls.h"
#include "RealizationOrder.h"
#include "ScheduleFunctions.h"

namespace Halide {
namespace Internal {

using std::string;
using std::vector;
using std::set;
using std::map;

namespace {
const bool trace_have_needs = false;
const bool trace_provides = false;
const bool trace_messages = false;
const bool trace_progress = false;

// Removes last token of the string, delimited by '.'
string remove_suffix(const string &str) {
    size_t lastdot = str.find_last_of(".");
    if (lastdot != std::string::npos) {
        return str.substr(0, lastdot);
    } else {
        return str;
    }
}

// Return first token of the string delimited by '.'
string first_token(const string &str) {
    size_t firstdot = str.find_first_of(".");
    if (firstdot != std::string::npos) {
        return str.substr(0, firstdot);
    } else {
        return str;
    }
}

// Return second token of the string delimited by '.'
string second_token(const string &str) {
    size_t firstdot = str.find_first_of(".");
    if (firstdot != std::string::npos) {
        size_t seconddot = str.find_first_of(".", firstdot + 1);
        if (seconddot != std::string::npos) {
            size_t len = seconddot - (firstdot + 1);
            return str.substr(firstdot + 1, len);
        }
    }
    return str;
}

// Return a string representation of the given box.
string box2str(const Box &b) {
    std::stringstream mins, maxs;
    mins << "(";
    maxs << "(";
    for (unsigned i = 0; i < b.size(); i++) {
        mins << b[i].min;
        maxs << b[i].max;
        if (i < b.size() - 1) {
            mins << ", ";
            maxs << ", ";
        }
    }
    mins << ")";
    maxs << ")";
    return mins.str() + " to " + maxs.str();
}

Box offset_box(const Box &b, const vector<Expr> &offset) {
    internal_assert(b.size() == offset.size());
    Box result(b.size());
    for (unsigned i = 0; i < b.size(); i++) {
        result[i] = Interval(b[i].min - offset[i], b[i].max - offset[i]);
    }
    return result;
}

Expr round_away_from_zero(Expr e) {
    return select(e < 0, floor(e), ceil(e));
}

class ReplaceVariables : public IRMutator {
    const Scope<Expr> &replacements;
public:
    ReplaceVariables(const Scope<Expr> &r) : replacements(r) {}

    using IRMutator::visit;
    void visit(const Variable *op) {
        IRMutator::visit(op);
        if (replacements.contains(op->name)) {
            expr = replacements.get(op->name);
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

// Computes the intersection of the two given boxes. Makes a "best
// effort" to determine if the boxes do not intersect, but if the box
// intervals have free variables, the intersection returned may be
// empty at runtime (when the variable values are known). Thus, any
// code using the intersection returned from this function must check
// for validity at runtime. For that reason this is encapsulated in a
// separate class so that the types may not be mixed.
class BoxIntersection {
private:
    Box _box;
    bool known_empty;
public:
    BoxIntersection() : known_empty(false) {}

    BoxIntersection(const Box &a, const Box &b) {
        internal_assert(a.size() == b.size());
        unsigned size = a.size();
        _box = Box(size);
        for (unsigned i = 0; i < size; i++) {
            if (is_positive_const(b[i].min - a[i].max)) known_empty = true;
            Expr dim_min = max(a[i].min, b[i].min);
            Expr dim_max = min(a[i].max, b[i].max);
            _box[i] = Interval(dim_min, dim_max);
        }
    }

    // Return an expression determining whether the intersection is
    // empty or not.
    Expr empty() const {
        internal_assert(_box.size() > 0);
        if (known_empty) {
            return const_true();
        } else {
            // If any dimension's min is greater than its max, the
            // intersection is empty.
            Expr e = GT::make(_box[0].min, _box[0].max);
            for (unsigned i = 1; i < _box.size(); i++) {
                e = Or::make(e, GT::make(_box[i].min, _box[i].max));
            }
            return e;
        }
    }

    const Box &box() const { return _box; }
};

// Helper class that wraps information common to Buffer and Parameter classes.
// Also provides a wrapper for Provide nodes which do not have buffer references.
class AbstractBuffer {
public:
    typedef enum { Halide, InputImage, OutputImage } BufferType;

    AbstractBuffer() : _dimensions(-1), _distributed(false) {}
    AbstractBuffer(Type type, BufferType btype, const string &name) :
        _type(type), _btype(btype), _name(name), _dimensions(-1), _distributed(false) {}
    AbstractBuffer(Type type, BufferType btype, const string &name, const Buffer &buffer) :
        _type(type), _btype(btype), _name(name) {
        internal_assert(btype == InputImage);
        internal_assert(buffer.defined());
        internal_assert(buffer.distributed());
        _distributed = buffer.distributed();
        _dimensions = buffer.dimensions();
        for (int i = 0; i < buffer.dimensions(); i++) {
            Expr min = buffer.local_min(i);
            Expr max = min + buffer.local_extent(i) - 1;
            _bounds.push_back(Interval(min, max));
            _shape.push_back(Interval(min, max));
        }
    }

    bool distributed() const {
        return _distributed;
    }

    void set_distributed() {
        _distributed = true;
    }

    int dimensions() const {
        internal_assert(_dimensions >= 0);
        return _dimensions;
    }

    void set_dimensions(int d) {
        _dimensions = d;
    }

    Expr extent(int dim) const {
        internal_assert(dim >= 0 && dim < (int)extents.size());
        return extents[dim];
    }

    void set_extent(int dim, Expr extent) {
        if (dim >= (int)extents.size()) {
            extents.resize(dim+1);
        }
        extents[dim] = extent;
    }

    Expr min(int dim) const {
        internal_assert(dim >= 0 && dim < (int)mins.size());
        return mins[dim];
    }

    void set_min(int dim, Expr min) {
        if (dim >= (int)mins.size()) {
            mins.resize(dim+1);
        }
        mins[dim] = min;
    }

    Type type() const {
        return _type;
    }

    BufferType buffer_type() const {
        return _btype;
    }

    bool is_image() const {
        return _btype == InputImage || _btype == OutputImage;
    }

    bool is_input_image() const {
        return _btype == InputImage;
    }

    bool is_output_image() const {
        return _btype == OutputImage;
    }

    void set_buffer_type(BufferType t) {
        _btype = t;
    }

    Expr elem_size() const {
        return _type.bytes();
    }

    const string &name() const {
        return _name;
    }

    string extended_name() const {
        //return is_image() ? _name + "_extended" : _name;
        return _name;
    }

    // Return the size of the given box in bytes according to the type
    // of this buffer.
    Expr size_of(const Box &b) const {
        internal_assert(b.size() > 0);
        Expr num_elems = 1;
        for (unsigned i = 0; i < b.size(); i++) {
            num_elems *= b[i].max - b[i].min + 1;
        }
        return num_elems * elem_size();
    }

    // Return the region (in parameterized global coordinates) that
    // represents the shape of this buffer. This can be different than
    // 'have' when a buffer is allocated larger than is actually
    // produced.
    const Box &shape() const {
        internal_assert(!_shape.empty()) << _name;
        return _shape;
    }

    // Return the region (in parameterized global coordinates)
    // produced of this buffer.
    const Box &have() const {
        internal_assert(!_bounds.empty()) << _name;
        return _bounds;
    }

    // Return the region (in parameterized global coordinates)
    // required of this buffer by the given function.
    const Box &need(const string &func) const {
        if (_need_bounds.find(func) == _need_bounds.end()) {
            internal_error << "Buffer " << name() << " is not needed by " << func << "\n";
        }
        return _need_bounds.at(func);
    }

    // Set the shape of this buffer.
    void set_shape(const Box &b) {
        internal_assert(_shape.empty());
        internal_assert(!is_image());
        internal_assert(_dimensions == -1);
        internal_assert(b.size() > 0);
        set_dimensions(b.size());
        _shape = Box(b.size());
        for (unsigned i = 0; i < b.size(); i++) {
            _shape[i] = Interval(b[i].min, b[i].max);
        }
    }

    // Set the region produced of this buffer.
    void set_have_bounds(const Box &b) {
        internal_assert(_bounds.empty());
        _bounds = Box(b.size());
        for (unsigned i = 0; i < b.size(); i++) {
            _bounds[i] = Interval(b[i].min, b[i].max);
        }
    }

    // Set the region required of this buffer by a function.
    void set_need_bounds(const string &func, const Box &b, bool is_update) {
        auto it = _need_bounds.find(func);
        if (it == _need_bounds.end()) {
            _need_bounds[func] = Box(b.size());
            for (unsigned i = 0; i < b.size(); i++) {
                _need_bounds[func][i] = Interval(b[i].min, b[i].max);
            }
        } else {
            // Update step accessing the same buffer as its pure
            // step. That's fine, but merge the 'need' region to
            // encompass both.
            internal_assert(is_update);
            merge_boxes(it->second, b);
            _need_bounds[func] = it->second;
        }
    }

    // Return a box in local coordinates (i.e. counting from mins of
    // 0) corresponding to the given global region.
    Box local_region(const Box &b, const string &func = "") const {
        Box result(b.size());
        Box local_origin;
        if (is_image()) {
            if (func.empty()) {
                local_origin = shape();
            } else {
                local_origin = need(func);
            }
        } else {
            local_origin = shape();
        }
        for (unsigned i = 0; i < b.size(); i++) {
            result[i] = Interval(b[i].min - local_origin[i].min, b[i].max - local_origin[i].min);
        }
        return result;
    }

    void merge_footprint(const vector<Expr> &fp) {
        if (_footprint.empty()) {
            _footprint.insert(_footprint.begin(), fp.begin(), fp.end());
        } else {
            internal_assert(fp.size() == _footprint.size());
            for (unsigned i = 0; i < fp.size(); i++) {
                _footprint[i] += max(0, fp[i]);
            }
        }
    }

    const vector<Expr> &footprint() const {
        return _footprint;
    }
private:
    Type _type;
    BufferType _btype;
    string _name;
    int _dimensions;
    bool _distributed;
    vector<Expr> mins;
    vector<Expr> extents;
    Box _bounds;
    Box _shape;
    map<string, Box> _need_bounds;
    vector<Expr> _footprint;
};

// Build a set of all the variables referenced. This traverses through
// any Let statements that are in the given environment, meaning both
// the "let" variable and the variables used to define it in the Let
// statement are in the resulting set.
class GetVariablesInExpr : public IRVisitor {
    using IRVisitor::visit;
    void visit(const Variable *var) {
        names.insert(var->name);
        if (env.contains(var->name)) {
            Expr value = env.get(var->name);
            value.accept(this);
        }
        IRVisitor::visit(var);
    }
public:
    const Scope<Expr> &env;
    set<string> names;
    GetVariablesInExpr(const Scope<Expr> &e) : env(e) {}
};

// Build a list of all input buffers using a particular variable as an
// index.
class FindBuffersUsingVariable : public IRVisitor {
    Scope<Expr> env;
    using IRVisitor::visit;

    void visit(const Let *let) {
        env.push(let->name, let->value);
        IRVisitor::visit(let);
        env.pop(let->name);
    }

    void visit(const LetStmt *let) {
        env.push(let->name, let->value);
        IRVisitor::visit(let);
        env.pop(let->name);
    }

    void visit(const Call *call) {
        GetVariablesInExpr vars(env);
        for (Expr arg : call->args) {
            arg.accept(&vars);
        }
        if (vars.names.count(name)) {
            // These are the only two call types that can be buffer references.
            if (call->call_type == Call::Image) {
                if (call->image.defined()) {
                    buffers.push_back(AbstractBuffer(call->image.type(), AbstractBuffer::InputImage, call->image.name(), (Buffer)call->image));
                } else {
                    buffers.push_back(AbstractBuffer(call->param.type(), AbstractBuffer::InputImage, call->param.name(), call->param.get_buffer()));
                }
            } else if (call->call_type == Call::Halide) {
                internal_assert(call->func.outputs() == 1);
                buffers.push_back(AbstractBuffer(call->func.output_types()[0], AbstractBuffer::Halide, call->func.name()));
            }

        }
        IRVisitor::visit(call);
    }

    void visit(const Provide *op) {
        GetVariablesInExpr vars(env);
        for (Expr arg : op->args) {
            arg.accept(&vars);
        }
        if (vars.names.count(name)) {
            internal_assert(op->values.size() == 1);
            buffers.push_back(AbstractBuffer(op->values[0].type(), AbstractBuffer::Halide, op->name));
        }
        IRVisitor::visit(op);
    }
public:
    string name;
    vector<AbstractBuffer> buffers;
    FindBuffersUsingVariable(string n) : name(n) {}
};

// Return a list of the buffers used in the given for loop.
map<string, AbstractBuffer> buffers_used(const For *for_loop) {
    FindBuffersUsingVariable find(for_loop->name);
    for_loop->body.accept(&find);
    vector<AbstractBuffer> buffers(find.buffers.begin(), find.buffers.end());
    map<string, AbstractBuffer> result;

    for (AbstractBuffer buf : buffers) {
        if (buf.is_image()) {
            internal_assert(!buf.have().empty());
        }
        result[buf.name()] = buf;
    }

    return result;
}

// Return the (symbolic) address of the given buffer at the given
// byte index.
Expr address_of(const string &buffer, Expr index) {
    Expr first_elem = Load::make(UInt(8), buffer, index, Buffer(), Parameter());
    return Call::make(Handle(), Call::address_of, {first_elem}, Call::Intrinsic);
}

// Return total number of processors available.
Expr num_processors() {
    return Call::make(Int(32), "halide_do_distr_size", {}, Call::Extern);
}

// Return rank of the current processor.
Expr rank() {
    return Call::make(Int(32), "halide_do_distr_rank", {}, Call::Extern);
}

// Insert call to communicate the given box of a buffer to 'rank'.
typedef enum { Send, Recv } CommunicateCmd;
Stmt communicate_subarray(CommunicateCmd cmd, const string &name,
                          Type t, const Box &shape, const Box &b, Expr rank) {
    vector<Expr> args;
    int ndims = (int)b.size();
    string size_buf = name + "_sizes",
        subsize_buf = name + "_subsizes",
        starts_buf = name + "_starts";

    /*
      Notes:

      - Because the arrays are row-major, dimension 0 corresponds to
        the outermost dimension, and dimension n-1 is the row
        dimension. This is the opposite of how Boxes and Bounds are
        stored in Halide, so we have to transpose here.
     */

    Expr address = address_of(name, 0);
    Stmt size_stores = Store::make(size_buf, shape[ndims - 1].max - shape[ndims - 1].min + 1, 0);
    Stmt subsize_stores = Store::make(subsize_buf, b[ndims - 1].max - b[ndims - 1].min + 1, 0);
    Stmt start_stores = Store::make(starts_buf, b[ndims - 1].min, 0);
    for (int i = ndims - 2, j=1; i >= 0; i--, j++) {
        Expr size = shape[i].max - shape[i].min + 1;
        Expr subsize = b[i].max - b[i].min + 1;
        Expr start = b[i].min;
        Stmt store_size = Store::make(size_buf, size, j);
        Stmt store_subsize = Store::make(subsize_buf, subsize, j);
        Stmt store_start = Store::make(starts_buf, start, j);
        size_stores = Block::make(size_stores, store_size);
        subsize_stores = Block::make(subsize_stores, store_subsize);
        start_stores = Block::make(start_stores, store_start);
    }

    Expr sizes = address_of(size_buf, 0),
        subsizes = address_of(subsize_buf, 0),
        starts = address_of(starts_buf, 0);
    args = {address, t.code, t.bits, ndims, sizes, subsizes, starts, rank};
    Expr call;
    switch (cmd) {
    case Send:
        call = Call::make(Int(32), "halide_do_distr_isend_subarray", args, Call::Extern);
        break;
    case Recv:
        call = Call::make(Int(32), "halide_do_distr_irecv_subarray", args, Call::Extern);
        break;
    }
    Stmt stores = Block::make(size_stores, Block::make(subsize_stores, start_stores));
    Stmt stmt = Block::make(stores, Evaluate::make(call));

    Stmt allocate = Allocate::make(size_buf, Int(32), {ndims}, const_true(), stmt);
    allocate = Allocate::make(subsize_buf, Int(32), {ndims}, const_true(), allocate);
    allocate = Allocate::make(starts_buf, Int(32), {ndims}, const_true(), allocate);

    return allocate;
}

// Insert call to send 'count' bytes starting at 'address' to 'rank'.
Expr send(Expr address, Type t, Expr count, Expr rank) {
    return Call::make(Int(32), "halide_do_distr_send", {address, t.code, t.bits, count, rank}, Call::Extern);
}

// Insert call to isend 'count' bytes starting at 'address' to 'rank'.
Expr isend(Expr address, Type t, Expr count, Expr rank) {
    return Call::make(Int(32), "halide_do_distr_isend", {address, t.code, t.bits, count, rank}, Call::Extern);
}

Stmt isend_subarray(const AbstractBuffer &buf, const Box &shape, const Box &b, Expr rank) {
    return communicate_subarray(Send, buf.name(), buf.type(), shape, b, rank);
}

// Insert call to receive 'count' bytes from 'rank' to buffer starting at 'address'.
Expr recv(Expr address, Type t, Expr count, Expr rank) {
    return Call::make(Int(32), "halide_do_distr_recv", {address, t.code, t.bits, count, rank}, Call::Extern);
}

// Insert call to irecv 'count' bytes from 'rank' to buffer starting at 'address'.
Expr irecv(Expr address, Type t, Expr count, Expr rank) {
    return Call::make(Int(32), "halide_do_distr_irecv", {address, t.code, t.bits, count, rank}, Call::Extern);
}

Stmt irecv_subarray(const AbstractBuffer &buf, const Box &shape, const Box &b, Expr rank) {
    return communicate_subarray(Recv, buf.extended_name(), buf.type(), shape, b, rank);
}

// Wait for all outstanding irecvs.
Stmt waitall_irecv(const string &name) {
    Expr address = address_of(name, 0);
    Expr rc = Call::make(Int(32), "halide_do_distr_waitall_recvs", {address}, Call::Extern);
    return Evaluate::make(rc);
}

// Wait for all outstanding isends.  The buffer name is unnecessary
// except to prevent Halide and/or LLVM optimizing away these
// calls. The argument is necessary to indicate that they may have
// side effects affecting the buffer, and therefore cannot be moved or
// optimized away.
Stmt waitall_isend(const string &name) {
    Expr address = address_of(name, 0);
    Expr rc = Call::make(Int(32), "halide_do_distr_waitall_sends", {address}, Call::Extern);
    return Evaluate::make(rc);
}

// Construct a statement to copy 'size' bytes from src to dest.
Stmt copy_memory(Expr dest, Expr src, Expr size) {
    return Evaluate::make(Call::make(UInt(8), Call::copy_memory,
                                    {dest, src, size}, Call::Intrinsic));
}

class ChangeDistributedLoopBuffers : public IRMutator {
    string name, newname;
    const Box &box;
    bool change_calls;
public:
    using IRMutator::visit;
    ChangeDistributedLoopBuffers(const string &n, const string &nn, const Box &b, bool c) :
        name(n), newname(nn), box(b), change_calls(c) {}

    void visit(const Call *call) {
        if (change_calls && call->name == name) {
            vector<Expr> newargs;
            for (unsigned i = 0; i < box.size(); i++) {
                newargs.push_back(call->args[i] - box[i].min);
            }
            expr = Call::make(call->type, newname, newargs,
                              call->call_type, call->func, call->value_index,
                              call->image, call->param);
        } else {
            IRMutator::visit(call);
        }
    }

    void visit(const Provide *provide) {
        if (!change_calls && provide->name == name) {
            vector<Expr> newargs;
            for (unsigned i = 0; i < box.size(); i++) {
                newargs.push_back(provide->args[i] - box[i].min);
            }
            Stmt newprovide = Provide::make(newname, provide->values, newargs);
            if (trace_provides) {
                Stmt p = Evaluate::make(print_when(rank() == 0, {string("rank"), rank(),
                                string("providing to"), provide->name,
                                string("global ["), provide->args[0], provide->args[1], string("],"),
                                string("local ["), newargs[0], newargs[1], string("] ="),
                                provide->values[0]}));
                newprovide = Block::make(p, newprovide);
            }
            stmt = newprovide;
        } else {
            IRMutator::visit(provide);
        }
    }
};

typedef enum { Pack, Unpack } PackCmd;
// Construct a statement to pack/unpack the given box of the given buffer
// to/from the contiguous scratch region of memory with the given name.
Stmt pack_region(PackCmd cmd, Type t, const string &scratch_name, const string &buffer_name, const Box &buffer_shape, const Box &b) {
    internal_assert(b.size() > 0);
    vector<Var> dims;
    for (unsigned i = 0; i < b.size(); i++) {
        dims.push_back(Var(buffer_name + "_dim" + std::to_string(i)));
    }

    // Construct src/dest pointer expressions as expressions in
    // terms of the box dimension variables.
    Expr bufferoffset = b[0].min * t.bytes(), scratchoffset = 0;
    Expr scratchstride = b[0].max - b[0].min + 1,
        bufferstride = buffer_shape[0].max - buffer_shape[0].min + 1;
    for (unsigned i = 1; i < b.size(); i++) {
        Expr extent = b[i].max - b[i].min + 1;
        Expr dim = dims[i];
        bufferoffset += (dim + b[i].min) * bufferstride * t.bytes();
        scratchoffset += dim * scratchstride * t.bytes();
        scratchstride *= extent;
        bufferstride *= buffer_shape[i].max - buffer_shape[i].min + 1;
    }

    // Construct loop nest to copy each contiguous row.  TODO:
    // ensure this nesting is in the correct row/column major
    // order.
    Expr rowsize = (b[0].max - b[0].min + 1) * t.bytes();
    Expr bufferaddr;
    Expr scratchaddr = address_of(scratch_name, scratchoffset);

    Stmt copyloop;
    switch (cmd) {
    case Pack:
        bufferaddr = address_of(buffer_name, bufferoffset);
        copyloop = copy_memory(scratchaddr, bufferaddr, rowsize);
        break;
    case Unpack:
        bufferaddr = address_of(buffer_name, bufferoffset);
        copyloop = copy_memory(bufferaddr, scratchaddr, rowsize);
        break;
    }
    //for (int i = b.size() - 1; i >= 1; i--) {
    for (int i = 1; i < (int)b.size(); i++) {
        copyloop = For::make(dims[i].name(), 0, b[i].max - b[i].min + 1,
                             ForType::Serial, DeviceAPI::Host, copyloop);
    }
    return copyloop;
}

Stmt copy_box(Type t, const string &src_buffer, const Box &src_shape, const Box &src_box,
              const string &dest_buffer, const Box &dest_shape, const Box &dest_box) {
    internal_assert(src_box.size() == dest_box.size());
    vector<Var> dims;
    for (unsigned i = 0; i < src_box.size(); i++) {
        dims.push_back(Var(src_buffer + "_dim" + std::to_string(i)));
    }

    // Construct src/dest pointer expressions as expressions in
    // terms of the box dimension variables.
    Expr destoffset = dest_box[0].min * t.bytes(),
        srcoffset = src_box[0].min * t.bytes();
    Expr srcstride = src_shape[0].max - src_shape[0].min + 1,
        deststride = dest_shape[0].max - dest_shape[0].min + 1;
    for (unsigned i = 1; i < dest_box.size(); i++) {
        Expr dim = dims[i];
        destoffset += (dim + dest_box[i].min) * deststride * t.bytes();
        deststride *= dest_shape[i].max - dest_shape[i].min + 1;
        srcoffset += (dim + src_box[i].min) * srcstride * t.bytes();
        srcstride *= src_shape[i].max - src_shape[i].min + 1;
    }

    // Construct loop nest to copy each contiguous row.  TODO:
    // ensure this nesting is in the correct row/column major
    // order.
    Expr rowsize = (dest_box[0].max - dest_box[0].min + 1) * t.bytes();
    Expr destaddr = address_of(dest_buffer, destoffset);
    Expr srcaddr = address_of(src_buffer, srcoffset);

    Stmt copyloop = copy_memory(destaddr, srcaddr, rowsize);
    // for (int i = dest_box.size() - 1; i >= 1; i--) {
    for (int i = 1; i < (int)dest_box.size(); i++) {
        copyloop = For::make(dims[i].name(), 0, dest_box[i].max - dest_box[i].min + 1,
                             ForType::Serial, DeviceAPI::Host, copyloop);
    }
    return copyloop;
}

// Allocate a buffer of the given name, type, and size that will
// be used in the given body.
Stmt allocate_scratch(const string &name, Type type, const Box &b, Stmt body) {
    vector<Expr> extents;
    Expr stride = 1;
    for (unsigned i = 0; i < b.size(); i++) {
        Expr extent = b[i].max - b[i].min + 1;
        body = LetStmt::make(name + ".min." + std::to_string(i), 0, body);
        body = LetStmt::make(name + ".stride." + std::to_string(i), stride, body);
        extents.push_back(extent);
        stride *= extent;
    }
    return Allocate::make(name, type, extents, const_true(), body);
}

// For each required region, copy the on-node portion into an
// extended buffer (big enough for required region including
// ghost zone).
Stmt copy_on_node_data(const string &func, const vector<AbstractBuffer> &required) {
    Stmt copy;
    for (const auto it : required) {
        const AbstractBuffer &in = it;
        const Box &have = in.have();
        const Box &need = it.need(func);

        // Only need to copy input images (not output images, which
        // don't have extended buffers).
        if (!in.is_input_image()) continue;

        BoxIntersection I(have, need);
        Box dest_box = in.local_region(I.box(), func);
        Box src_box = in.local_region(I.box());
        Var numbytes = Var("numbytes");
        Stmt s;
        if (I.box().size() == 1) {
            Expr destoff = dest_box[0].min, srcoff = src_box[0].min;
            Expr destoffbytes = destoff * in.elem_size(), srcoffbytes = srcoff * in.elem_size();
            Expr dest = address_of(in.extended_name(), destoffbytes), src = address_of(in.name(), srcoffbytes);
            s = copy_memory(dest, src, numbytes);
        } else {
            s = copy_box(in.type(), in.name(), in.shape(), src_box, in.extended_name(), need, dest_box);
        }

        Expr cond = GT::make(numbytes, 0);
        s = IfThenElse::make(cond, s);
        s = LetStmt::make(numbytes.name(), in.size_of(dest_box), s);
        if (copy.defined()) {
            copy = Block::make(copy, s);
        } else {
            copy = s;
        }
    }
    return copy;
}

// Generate communication code to send/recv the intersection of the
// 'have' and 'need' regions of 'buf'.
Stmt communicate_intersection(CommunicateCmd cmd, const AbstractBuffer &buf, const string &func) {
    const Box &have = buf.have();
    const Box &need = buf.need(func);

    Scope<Expr> env;
    env.push("Rank", Var("r"));
    Box have_parameterized = simplify_box(have, env);
    Box need_parameterized = simplify_box(need, env);
    BoxIntersection I;

    switch (cmd) {
    case Send:
        I = BoxIntersection(have, need_parameterized);
        break;
    case Recv:
        I = BoxIntersection(have_parameterized, need);
        break;
    }

    Expr addr;
    Expr numbytes = Var("msgsize");
    Expr cond = And::make(NE::make(Var("Rank"), Var("r")), GT::make(numbytes, 0));
    Stmt commstmt;

    // Convert the intersection box to "local" coordinates (the
    // extended buffer counts from 0). This just means subtracting the
    // min global coordinate from the intersection bounds (which are
    // also global) to get a local coordinate starting from 0.
    Box local_have = buf.local_region(I.box()),
        local_need = buf.local_region(I.box(), func);

    switch (cmd) {
    case Send:
        if (local_have.size() == 1) {
            addr = address_of(buf.name(), local_have[0].min * buf.elem_size());
            Expr msgsize = local_have[0].max - local_have[0].min + 1;
            commstmt = IfThenElse::make(cond, Evaluate::make(isend(addr, buf.type(), msgsize, Var("r"))));
        } else {
            commstmt = isend_subarray(buf, buf.shape(), local_have, Var("r"));
        }
        break;
    case Recv:
        if (local_need.size() == 1) {
            addr = address_of(buf.extended_name(), local_need[0].min * buf.elem_size());
            Expr msgsize = local_have[0].max - local_have[0].min + 1;
            commstmt = IfThenElse::make(cond, Evaluate::make(irecv(addr, buf.type(), msgsize, Var("r"))));
        } else {
            Box shape = buf.is_image() ? need : buf.shape();
            commstmt = irecv_subarray(buf, shape, local_need, Var("r"));
        }
        break;
    }
    Box shape = buf.shape();
    if (trace_messages && cmd == Send) {
        Stmt p = Evaluate::make(print_when(cond, {string("rank"), rank(),
                        string("sending to rank"), Var("r"),
                        string("buffer " + buf.name() + ":\n"),
                        string("size"),
                        I.box()[0].max - I.box()[0].min + 1,
                        string("x"), I.box()[1].max - I.box()[1].min + 1, string("\n"),
                        string("\n   shape[0].min ="), shape[0].min, string("shape[0].max ="), shape[0].max,
                        string("\n   shape[1].min ="), shape[1].min, string("shape[1].max ="), shape[1].max,
                        string("\n   have[0].min ="), have[0].min, string("have[0].max ="), have[0].max,
                        string("\n   have[1].min ="), have[1].min, string("have[1].max ="), have[1].max,
                        string("\n   need_parameterized[0].min ="), need_parameterized[0].min, string("need_parameterized[0].max ="), need_parameterized[0].max,
                        string("\n   need_parameterized[1].min ="), need_parameterized[1].min, string("need_parameterized[1].max ="), need_parameterized[1].max,
                        string("\n   I.box()[0].min ="), I.box()[0].min, string("I.box()[0].max ="), I.box()[0].max,
                        string("\n   I.box()[1].min ="), I.box()[1].min, string("I.box()[1].max ="), I.box()[1].max,
                        string("\n   local_have[0].min ="), local_have[0].min, string("local_have[0].max ="), local_have[0].max,
                        string("\n   local_have[1].min ="), local_have[1].min, string("local_have[1].max ="), local_have[1].max
                        }));
        commstmt = Block::make(p, commstmt);
    }

    if (trace_messages && cmd == Recv) {
        Stmt p = Evaluate::make(print_when(cond, {string("rank"), rank(),
                        string("receiving from rank"), Var("r"),
                        string("buffer " + buf.name() + ":\n"),
                        string("size"),
                        I.box()[0].max - I.box()[0].min + 1,
                        string("x"), I.box()[1].max - I.box()[1].min + 1, string("\n"),
                        string("\n   shape[0].min ="), shape[0].min, string("shape[0].max ="), shape[0].max,
                        string("\n   shape[1].min ="), shape[1].min, string("shape[1].max ="), shape[1].max,
                        string("\n   have_parameterized[0].min ="), have_parameterized[0].min, string("have_parameterized[0].max ="), have_parameterized[0].max,
                        string("\n   have_parameterized[1].min ="), have_parameterized[1].min, string("have_parameterized[1].max ="), have_parameterized[1].max,
                        string("\n   need[0].min ="), need[0].min, string("need[0].max ="), need[0].max,
                        string("\n   need[1].min ="), need[1].min, string("need[1].max ="), need[1].max,
                        string("\n   I.box()[0].min ="), I.box()[0].min, string("I.box()[0].max ="), I.box()[0].max,
                        string("\n   I.box()[1].min ="), I.box()[1].min, string("I.box()[1].max ="), I.box()[1].max,
                        string("\n   local_need[0].min ="), local_need[0].min, string("local_need[0].max ="), local_need[0].max,
                        string("\n   local_need[1].min ="), local_need[1].min, string("local_need[1].max ="), local_need[1].max
                        }));
        commstmt = Block::make(p, commstmt);
    }

    commstmt = IfThenElse::make(cond, commstmt);
    commstmt = LetStmt::make("msgsize", buf.size_of(I.box()), commstmt);
    commstmt = For::make("r", 0, Var("NumProcessors"), ForType::Serial, DeviceAPI::Host, commstmt);

    return commstmt;
}

// For each required region, generate communication code between ranks
// that own data needed by other ranks.
Stmt exchange_data(const string &func, const vector<AbstractBuffer> &required) {
    Stmt sendloop, recvloop;
    Stmt sendwait, recvwait;
    for (const auto it : required) {
        const AbstractBuffer &in = it;
        // No border exchanges needed for non-distributed buffers or
        // output images.
        if (!in.distributed() || in.is_output_image()) continue;

        if (sendloop.defined()) {
            sendloop = Block::make(sendloop, communicate_intersection(Send, in, func));
            sendwait = Block::make(sendwait, waitall_isend(in.name()));
        } else {
            sendloop = communicate_intersection(Send, in, func);
            sendwait = waitall_isend(in.name());
        }

        if (recvloop.defined()) {
            recvloop = Block::make(recvloop, communicate_intersection(Recv, in, func));
            recvwait = Block::make(recvwait, waitall_irecv(in.extended_name()));
        } else {
            recvloop = communicate_intersection(Recv, in, func);
            recvwait = waitall_irecv(in.extended_name());
        }
    }
    internal_assert(sendloop.defined() == recvloop.defined());
    if (!sendloop.defined()) {
        return Stmt();
    } else {
        Stmt comm = Block::make(recvloop, sendloop);
        Stmt wait = Block::make(recvwait, sendwait);
        return Block::make(comm, wait);
    }
}

// Change all uses of the original input buffers to use the extended
// buffers, and modify output buffer indices to be "local" indices
// (instead of global).
Stmt update_io_buffers(Stmt loop, const string &func, const vector<AbstractBuffer> &required,
                       const vector<AbstractBuffer> &provided) {
    for (const auto it : required) {
        const AbstractBuffer &in = it;
        const Box &b = in.need(func);
        if (!in.is_image()) continue;
        if (in.is_input_image()) {
            ChangeDistributedLoopBuffers change(in.name(), in.extended_name(), b, true);
            loop = change.mutate(loop);
        } else {
            // Note that output images being used as inputs do not
            // have an extended buffer, but we still need to correct
            // the indices.
            internal_assert(in.is_output_image());
            ChangeDistributedLoopBuffers change(in.name(), in.name(), b, true);
            loop = change.mutate(loop);
        }
    }

    for (const auto it : provided) {
        const AbstractBuffer &out = it;
        const Box &b = out.have();
        if (!out.is_image()) continue;
        ChangeDistributedLoopBuffers change(out.name(), out.name(), b, false);
        loop = change.mutate(loop);
    }
    return loop;
}

// Allocate extended buffers for the given body.
Stmt allocate_extended_buffers(Stmt body, const string &func, const vector<AbstractBuffer> &required) {
    Stmt allocates = body;
    for (const auto it : required) {
        const AbstractBuffer &in = it;
        const Box &b = in.need(func);
        if (!in.is_input_image()) continue;
        allocates = allocate_scratch(in.extended_name(), in.type(), b, allocates);
    }
    return allocates;
}

}

class InjectCommunication : public IRMutator {
    Stmt inject_communication(const string &name, Stmt s) const {
        map<string, Box> r, p;
        vector<AbstractBuffer> required, provided;
        Stmt newstmt = s;

        r = boxes_required(s);
        p = boxes_provided(s);

        for (auto it : r) {
            internal_assert(buffers.find(it.first) != buffers.end());
            required.push_back(buffers.at(it.first));
        }
        for (auto it : p) {
            internal_assert(buffers.find(it.first) != buffers.end()) << it.first;
            provided.push_back(buffers.at(it.first));
        }

        // Stmt copy = copy_on_node_data(name, required);
        // if (copy.defined()) {
        //     newstmt = Block::make(copy, newstmt);
        // }
        if (trace_progress) {
            Stmt p = Evaluate::make(print({string("rank"), rank(), string("stage"), name,
                            string("before copy_on_node_data")}));
            newstmt = Block::make(p, newstmt);
            p = Evaluate::make(print({string("rank"), rank(), string("stage"), name,
                            string("after copy_on_node_data")}));
            newstmt = Block::make(newstmt, p);
        }

        Stmt border_exchange = exchange_data(name, required);
        if (border_exchange.defined()) {
            // TODO: move the isend waitall after computation.
            newstmt = Block::make(border_exchange, newstmt);
        }

        // if (trace_progress) {
        //     Stmt p = Evaluate::make(print({string("rank"), rank(), string("stage"), name,
        //                     string("before exchange_data")}));
        //     newstmt = Block::make(p, newstmt);
        //     p = Evaluate::make(print({string("rank"), rank(), string("stage"), name,
        //                     string("after exchange_data")}));
        //     newstmt = Block::make(newstmt, p);
        // }

        newstmt = update_io_buffers(newstmt, name, required, provided);
        // newstmt = allocate_extended_buffers(newstmt, name, required);

        if (trace_have_needs) {
            Stmt p;
            for (const auto &in : required) {
                Box have = in.have(), need = in.need(name);
                if (in.name() == "f") continue;
                Stmt pp = Evaluate::make(print_when(rank() == 1, {string("rank"), rank(),
                                string("function " + name + " requires"),
                                string("buffer " + in.name() + ":\n"),
                                string("have size"),
                                have[0].max - have[0].min + 1, string("x"), have[1].max - have[1].min + 1, string("\n"),
                                string("need size (" + in.extended_name() + ")"),
                                need[0].max - need[0].min + 1, string("x"), need[1].max - need[1].min + 1, string("\n"),
                                string("\n   have[0].min ="), have[0].min, string("have[0].max ="), have[0].max,
                                string("\n   have[1].min ="), have[1].min, string("have[1].max ="), have[1].max,
                                string("\n   need[0].min ="), need[0].min, string("need[0].max ="), need[0].max,
                                string("\n   need[1].min ="), need[1].min, string("need[1].max ="), need[1].max
                                }));
                if (p.defined()) {
                    p = Block::make(p, pp);
                } else {
                    p = pp;
                }
            }
            newstmt = Block::make(p, newstmt);
        }

        return newstmt;
    }

public:
    const map<string, AbstractBuffer> &buffers;
    InjectCommunication(const map<string, AbstractBuffer> &bufs) : buffers(bufs) {}

    using IRMutator::visit;

    void visit(const ProducerConsumer *op) {
        string current_function = op->name;
        Stmt newproduce = inject_communication(op->name, op->produce);
        stmt = ProducerConsumer::make(op->name, newproduce,
                                      op->update.defined() ? inject_communication(op->name, op->update) : mutate(op->update),
                                      mutate(op->consume));
    }
};

// For each distributed for loop, mutate its bounds to be determined
// by processor rank.
class DistributeLoops : public IRMutator {
public:
    set<string> slice_size_inserted;
    const map<string, Expr> &distributed_bounds;
    const std::map<std::string, Function> &env;
    bool cap_extents;
    DistributeLoops(const map<string, Expr> &bounds, const std::map<std::string, Function> &e, bool cap=false) : distributed_bounds(bounds), env(e), cap_extents(cap) {}

    using IRMutator::visit;
    void visit(const LetStmt *let) {
        if (distributed_bounds.find(let->name) != distributed_bounds.end()) {
            string loop_var = remove_suffix(let->name);
            string funcname = first_token(let->name);
            string stage_prefix = funcname + "." + second_token(let->name);
            Expr oldmin = distributed_bounds.at(loop_var + ".loop_min"),
                oldmax = distributed_bounds.at(loop_var + ".loop_max"),
                oldextent = distributed_bounds.at(loop_var + ".loop_extent");
            Expr slice_size = cast(Int(32), ceil(cast(Float(32), oldextent) / Var("NumProcessors")));

            // Check if this dimension was fused, and get the inner
            // extent if so.
            Expr inner;
            for (Split s : env.at(funcname).schedule().splits()) {
                if (s.is_fuse() && ends_with(loop_var, s.old_var)) {
                    internal_assert(!inner.defined());
                    Var inner_extent(stage_prefix + "." + s.inner + ".loop_extent");
                    inner = inner_extent;
                }
            }

            // If the dimension was fused, we have to round up our
            // slice size to be a multiple of the inner
            // dimension. This is so that distributing fused
            // dimensions maintains the invariant that each processor
            // gets an axis-aligned bounding box of the buffer in
            // question. Without rounding up, you can have a situation
            // e.g. with tiling where distributing a fused tile
            // dimension splits up the input buffer among processor
            // ranks non axis-aligned.
            if (inner.defined()) {
                Expr numrows = (slice_size + inner - 1) / inner;
                slice_size = numrows * inner;
            }

            Expr newmin = oldmin + Var(loop_var + ".SliceSize") * Var("Rank"),
                newmax = newmin + Var(loop_var + ".SliceSize") - 1;
            // We by default don't cap the new extent to make sure it
            // doesn't run over. That is because allocation bounds
            // inference will allocate a buffer big enough for the
            // entire slice, meaning the accesses will not be out of
            // bounds, just full of garbage. The only time we cap the
            // extents is for DistributedImage in order to know the
            // non-garbage local extents.
            Expr newextent = (cap_extents ? min(newmax, oldmax) : newmax) - newmin + 1;
            bool insert_sz = !slice_size_inserted.count(loop_var);
            slice_size_inserted.insert(loop_var);
            if (ends_with(let->name, ".loop_min")) {
                stmt = LetStmt::make(let->name, newmin, mutate(let->body));
            } else if (ends_with(let->name, ".loop_max")) {
                stmt = LetStmt::make(let->name, newmax, mutate(let->body));
            } else if (ends_with(let->name, ".loop_extent")) {
                stmt = LetStmt::make(let->name, newextent, mutate(let->body));
            } else {
                internal_assert(false) << let->name;
            }
            if (insert_sz) {
                stmt = LetStmt::make(loop_var + ".SliceSize", slice_size, stmt);
                Var np = Var("NumProcessors");
                Expr error = Call::make(Int(32), "halide_error_dim_over_distributed",
                                        {loop_var, oldextent, np},
                                        Call::Extern);
                Stmt assert = AssertStmt::make(oldextent >= np, error);
                stmt = Block::make(assert, stmt);
            }
        } else {
            IRMutator::visit(let);
        }
    }
};

// Remove the "distributed" attribute from all distributed loops, and
// replace with the non-distributed serial/parallel version.
class ChangeDistributedFor : public IRMutator {
public:
    using IRMutator::visit;
    void visit(const For *for_loop) {
        IRMutator::visit(for_loop);
        if (for_loop->for_type == ForType::Distributed) {
            stmt = For::make(for_loop->name, for_loop->min, for_loop->extent,
                             ForType::Serial, for_loop->device_api,
                             for_loop->body);
        } else if (for_loop->for_type == ForType::DistributedParallel) {
            stmt = For::make(for_loop->name, for_loop->min, for_loop->extent,
                             ForType::Parallel, for_loop->device_api,
                             for_loop->body);
        }
    }
};

class FindDistributedLoops : public IRVisitor {
public:
    set<string> distributed_functions;
    map<string, Expr> distributed_bounds;

    using IRVisitor::visit;

    void visit(const LetStmt *let) {
        env.push(let->name, let->value);
        IRVisitor::visit(let);
        env.pop(let->name);
    }

    void visit(const For *for_loop) {
        if (for_loop->for_type == ForType::Distributed ||
            for_loop->for_type == ForType::DistributedParallel) {
            for (auto it = env.begin(), ite = env.end(); it != ite; ++it) {
                string prefix = for_loop->name + ".";
                if (starts_with(it.name(), prefix)) {
                    distributed_bounds[it.name()] = it.value();
                }
            }
            string funcname = first_token(for_loop->name);
            distributed_functions.insert(funcname);
        }
        IRVisitor::visit(for_loop);
    }
private:
    Scope<Expr> env;
};

// Construct a map of all input buffers used in a pipeline. The
// results are a map from buffer name -> AbstractBuffer with
// information about the buffer.
class GetPipelineBuffers : public IRVisitor {
public:
    map<string, AbstractBuffer> buffers;
    const set<string> &distributed_functions;
    GetPipelineBuffers(const set<string> &d) : distributed_functions(d) {}

    using IRVisitor::visit;

    void visit(const For *for_loop) {
        string funcname = first_token(for_loop->name);
        map<string, AbstractBuffer> bufs;
        bufs = buffers_used(for_loop);
        for (auto &it : bufs) {
            if (distributed_functions.count(it.first)) {
                it.second.set_distributed();
            }
        }
        buffers.insert(bufs.begin(), bufs.end());
        IRVisitor::visit(for_loop);
    }
};

// Set the bounds for all non-Image input buffers based on the region
// provided by their producing loops. This should take place *after*
// loops have been distributed, otherwise the bounds set will be
// global values.
class SetBufferBounds : public IRGraphVisitor {
    Scope<Expr> shallow_env;

    void set_bounds(const string &name, Stmt s, bool is_update = false) {
        map<string, Box> required = boxes_required(s);
        Box provided = box_provided(s, name);
        internal_assert(buffers.find(name) != buffers.end());
        AbstractBuffer &buf = buffers.at(name);
        internal_assert(!buf.is_image());
        if (!is_update) {
            // "have" bounds for an update will by definition be the
            // same as the pure stage, so we don't need to do anything
            // here.
            buf.set_have_bounds(simplify_box(provided, env));
        }

        for (auto it : required) {
            if (buffers.find(it.first) != buffers.end()) {
                AbstractBuffer &buf = buffers.at(it.first);
                buf.set_need_bounds(name, simplify_box(it.second, env), is_update);
            }
        }
    }
public:
    Scope<Expr> env;
    map<string, AbstractBuffer> &buffers;
    SetBufferBounds(map<string, AbstractBuffer> &bufs) : buffers(bufs) {}

    using IRGraphVisitor::visit;

    void visit(const LetStmt *let) {
        // TODO: Revisit why maintaining this environment is
        // necessary. There must be a better way to accomplish symbol
        // capture so the have/needs are in terms of globals, not
        // variables local to a production.
        shallow_env.push(let->name, let->value);
        Expr rhs = ReplaceVariables(shallow_env).mutate(let->value);
        env.push(let->name, rhs);
        IRGraphVisitor::visit(let);
        env.pop(let->name);
        shallow_env.pop(let->name);
    }

    void visit(const Let *let) {
        env.push(let->name, let->value);
        IRGraphVisitor::visit(let);
        env.pop(let->name);
    }

    void visit(const Realize *op) {
        AbstractBuffer &buf = buffers.at(op->name);
        internal_assert(!buf.is_image());
        Box b = box_touched(op->body, op->name);
        buf.set_shape(b);
        IRGraphVisitor::visit(op);
    }

    void visit(const ProducerConsumer *op) {
        set_bounds(op->name, op->produce);
        if (op->update.defined()) set_bounds(op->name, op->update, true);

        AbstractBuffer &buf = buffers.at(op->name);
        internal_assert(!buf.is_image());
        // We say that a producer with no consumer (i.e. the end of
        // the pipeline) is an output image, because we will need to
        // manually correct load/store indices. Note that output
        // images can be used as input buffers, if the last stage in
        // the pipeline has an update step.
        if (is_no_op((Stmt)op->consume)) {
            buf.set_buffer_type(AbstractBuffer::OutputImage);
        }

        IRGraphVisitor::visit(op);
    }
};

class SetBufferFootprints : public IRGraphVisitor {
public:
    map<string, AbstractBuffer> &buffers;
    SetBufferFootprints(map<string, AbstractBuffer> &bufs) : buffers(bufs) {}

    using IRGraphVisitor::visit;

    void visit(const Call *op) {
        map<string, Box> required = boxes_required(op);
        for (auto it : required) {
            if (buffers.find(it.first) != buffers.end()) {
                AbstractBuffer &buf = buffers.at(it.first);
                vector<Expr> fp;
                for (unsigned i = 0; i < it.second.size(); i++) {
                    fp.push_back(simplify(it.second[i].max - it.second[i].min + 1));
                }
                buf.merge_footprint(fp);
            }
        }
        IRGraphVisitor::visit(op);
    }
};

Stmt distribute_loops_only(Stmt s, const std::map<std::string, Function> &env, bool cap_extents) {
    FindDistributedLoops find;
    s.accept(&find);
    if (find.distributed_functions.empty()) {
        return s;
    }
    return DistributeLoops(find.distributed_bounds, env, cap_extents).mutate(s);
}

Stmt distribute_loops(Stmt s, const std::map<std::string, Function> &env) {
    FindDistributedLoops find;
    s.accept(&find);
    if (find.distributed_functions.empty()) {
        return s;
    }
    return DistributeLoops(find.distributed_bounds, env).mutate(s);
}

Stmt inject_communication(Stmt s, const std::map<std::string, Function> &env) {
    FindDistributedLoops find;
    s.accept(&find);
    if (find.distributed_functions.empty()) {
        return s;
    }
    GetPipelineBuffers getio(find.distributed_functions);
    s.accept(&getio);
    SetBufferBounds setb(getio.buffers);
    s.accept(&setb);
    SetBufferFootprints setfp(getio.buffers);
    s.accept(&setfp);
    s = InjectCommunication(getio.buffers).mutate(s);
    s = ChangeDistributedFor().mutate(s);
    s = LetStmt::make("Rank", rank(), s);
    s = LetStmt::make("NumProcessors", num_processors(), s);
    return s;
}

// -------------------------------------------------- Testing specific code:

namespace {
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
        map<string, Box> r = boxes_required(op), p = boxes_provided(op);
        for (auto it : r) {
            required[it.first] = simplify_box(it.second, env);
        }
        for (auto it : p) {
            provided[it.first] = simplify_box(it.second, env);
        }
    }

    map<string, Box> required, provided;
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
    FindDistributedLoops find;
    s.accept(&find);
    s = DistributeLoops(find.distributed_bounds, env).mutate(s);
    s = bounds_inference(s, outputs, order, env, func_bounds);
    s = allocation_bounds_inference(s, env, func_bounds);
    s = uniquify_variable_names(s);
    return s;
}

map<string, Box> func_boxes_provided(Func f) {
    Stmt s = partial_lower(f);
    GetBoxes get;
    s.accept(&get);
    return get.provided;
}

map<string, Box> func_boxes_required(Func f) {
    Stmt s = partial_lower(f);
    GetBoxes get;
    s.accept(&get);
    return get.required;
}

map<string, AbstractBuffer> func_input_buffers(Func f) {
    map<string, Function> env;
    Stmt s = partial_lower(f);
    FindDistributedLoops find;
    s.accept(&find);
    GetPipelineBuffers getio(find.distributed_functions);
    s.accept(&getio);
    SetBufferBounds setb(getio.buffers);
    s.accept(&setb);
    return getio.buffers;
}

int expr2int(Expr e) {
    const int *result = as_const_int(simplify(e));
    internal_assert(result != NULL) << e;
    return *result;
}

bool operator==(const Interval &a, const Interval &b) {
    int amin = expr2int(a.min), amax = expr2int(a.max);
    int bmin = expr2int(b.min), bmax = expr2int(b.max);
    return amin == bmin && amax == bmax;
}

}

void distribute_loops_test() {
    const int w = 20;
    const int numprocs = 2;
    Func clamped("clamped");
    Var x("x");
    DistributedImage<int> in(w, "in");
    in.set_domain(x);
    in.placement().distribute(x);
    in.allocate();
    clamped(x) = in(clamp(x, 0, w-1));

    {
        Func f("f");
        f(x) = clamped(x) + clamped(x+1);
        f.compute_root().distribute(x);

        map<string, AbstractBuffer> buffers = func_input_buffers(f);

        internal_assert(buffers.find(in.name()) != buffers.end());
        const AbstractBuffer &buf = buffers.at(in.name());
        const Box &have = buf.have();
        const Box &need = buf.need(f.name());

        Scope<Expr> testenv;
        testenv.push("Rank", 0);
        testenv.push("NumProcessors", numprocs);
        testenv.push(f.name() + ".min.0", 0);
        testenv.push(f.name() + ".max.0", w-1);
        testenv.push(f.name() + ".extent.0", w);

        {
            testenv.ref("Rank") = 0;
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(0, 9));
            internal_assert(need_concrete[0] == Interval(0, 10));
        }
        {
            testenv.ref("Rank") = 1;
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(10, 19));
            internal_assert(need_concrete[0] == Interval(10, 19));
        }
        {
            testenv.ref("Rank") = 1;
            Box have_concrete = simplify_box(have, testenv);
            BoxIntersection TI(have_concrete, need);
            testenv.ref("Rank") = 0;
            // What rank 1 has and rank 0 needs (index 10):
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(10, 10));
            internal_assert(expr2int(buf.size_of(intersection)) == 4);
        }
        {
            testenv.ref("Rank") = 0;
            Box have_concrete = simplify_box(have, testenv);
            BoxIntersection TI(have_concrete, need);
            testenv.ref("Rank") = 1;
            // What rank 0 has and rank 1 needs (nothing):
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(10, 9));
            internal_assert(expr2int(buf.size_of(intersection)) == 0);
        }
    }

    {
        Func f("f");
        f(x) = in(x) + 1;
        f.compute_root().distribute(x);
        map<string, AbstractBuffer> buffers = func_input_buffers(f);

        internal_assert(buffers.find(in.name()) != buffers.end());
        const AbstractBuffer &buf = buffers.at(in.name());
        const Box &b = buf.have();
        const Box &req = buf.need(f.name());

        Scope<Expr> testenv;
        testenv.push("Rank", Var("r"));
        testenv.push("NumProcessors", numprocs);
        testenv.push(f.name() + ".min.0", 0);
        testenv.push(f.name() + ".max.0", w-1);
        testenv.push(f.name() + ".extent.0", w);

        Box need = simplify_box(req, testenv);
        testenv.pop("Rank");
        Box have = simplify_box(b, testenv);
        testenv.push("Rank", 0);

        {
            testenv.ref("Rank") = 0;
            testenv.push("r", 0);
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(0, 9));
            internal_assert(need_concrete[0] == Interval(0, 9));
        }
        {
            testenv.ref("Rank") = 1;
            testenv.ref("r") = 1;
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(10, 19));
            internal_assert(need_concrete[0] == Interval(10, 19));
        }
        {
            testenv.ref("Rank") = 1;
            testenv.ref("r") = 0;
            BoxIntersection TI(have, need);
            // What rank 1 has and rank 0 needs (nothing):
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(10, 9));
            internal_assert(expr2int(buf.size_of(intersection)) == 0);
        }
        {
            BoxIntersection TI(have, need);
            testenv.ref("Rank") = 0;
            testenv.ref("r") = 1;
            // What rank 0 has and rank 1 needs (nothing):
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(10, 9));
            internal_assert(expr2int(buf.size_of(intersection)) == 0);
        }
    }

    {
        Func f("f"), g("g");
        f(x) = clamped(x) + clamped(x+1);
        g(x) = f(x) + f(x+1);
        f.compute_root().distribute(x);
        g.compute_root().distribute(x);

        map<string, AbstractBuffer> buffers = func_input_buffers(g);

        internal_assert(buffers.find(f.name()) != buffers.end());
        const AbstractBuffer &buf = buffers.at(f.name());
        const Box &b = buf.have();
        const Box &req = buf.need(g.name());

        Scope<Expr> testenv;
        testenv.push("Rank", Var("r"));
        testenv.push("NumProcessors", numprocs);

        testenv.push(g.name() + ".min.0", 0);
        testenv.push(g.name() + ".max.0", w-1);
        testenv.push(g.name() + ".extent.0", w);

        testenv.push(f.name() + ".min.0", Var(g.name() + ".min.0"));
        testenv.push(f.name() + ".max.0", Var(g.name() + ".max.0") + 1);
        testenv.push(f.name() + ".extent.0", Var(f.name() + ".max.0") - Var(f.name() + ".min.0") + 1);

        // First test have/need of the f buffer to function g.
        Box need = simplify_box(req, testenv);
        testenv.pop("Rank");
        Box have = simplify_box(b, testenv);
        testenv.push("Rank", 0);

        {
            testenv.ref("Rank") = 0;
            testenv.push("r", 0);
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(0, 10));
            internal_assert(need_concrete[0] == Interval(0, 10));
        }
        {
            testenv.ref("Rank") = 1;
            testenv.ref("r") = 1;
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(11, 21));
            internal_assert(need_concrete[0] == Interval(10, 20));
        }
        {
            testenv.ref("Rank") = 1;
            testenv.ref("r") = 0;
            BoxIntersection TI(have, need);
            // What rank 1 has and rank 0 needs (nothing):
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(11, 10));
            internal_assert(expr2int(buf.size_of(intersection)) == 0);
        }
        {
            BoxIntersection TI(have, need);
            testenv.ref("Rank") = 0;
            testenv.ref("r") = 1;
            // What rank 0 has and rank 1 needs (index 10):
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(10, 10));
            internal_assert(expr2int(buf.size_of(intersection)) == 4);

            // Local intersection for rank 0 is index 10:
            vector<Expr> offset;
            for (unsigned i = 0; i < have.size(); i++) {
                offset.push_back(have[i].min);
            }
            Box offsetI = offset_box(intersection, offset);
            Box localI = simplify_box(offsetI, testenv);
            internal_assert(localI[0] == Interval(10, 10));

            // Local intersection for rank 1 is index 0:
            for (unsigned i = 0; i < need.size(); i++) {
                offset[i] = need[i].min;
            }
            offsetI = offset_box(intersection, offset);
            localI = simplify_box(offsetI, testenv);
            internal_assert(localI[0] == Interval(0, 0));
        }

        // Now test have/need of the input buffer to function f.
        testenv.pop("r");
        testenv.ref("Rank") = Var("r");
        internal_assert(buffers.find(in.name()) != buffers.end());
        need = simplify_box(buffers.at(in.name()).need(f.name()), testenv);
        testenv.pop("Rank");
        have = simplify_box(buffers.at(in.name()).have(), testenv);
        testenv.push("Rank", 0);

        {
            testenv.ref("Rank") = 0;
            testenv.push("r", 0);
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(0, 9));
            internal_assert(need_concrete[0] == Interval(0, 11));
        }
        {
            testenv.ref("Rank") = 1;
            testenv.ref("r") = 1;
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(10, 19));
            // Note the overlap: this means both rank 0 and rank 1
            // need index 11 of the input.
            internal_assert(need_concrete[0] == Interval(11, 19));
        }
        {
            testenv.ref("Rank") = 1;
            testenv.ref("r") = 0;
            BoxIntersection TI(have, need);
            // What rank 1 has and rank 0 needs (index 10 and 11).
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(10, 11));
            internal_assert(expr2int(buf.size_of(intersection)) == 8);

            // Local intersection for rank 1 is index 0 and 1:
            vector<Expr> offset;
            for (unsigned i = 0; i < have.size(); i++) {
                offset.push_back(have[i].min);
            }
            Box offsetI = offset_box(intersection, offset);
            Box localI = simplify_box(offsetI, testenv);
            internal_assert(localI[0] == Interval(0, 1));

            // Local intersection for rank 0 is index 10 and 11:
            for (unsigned i = 0; i < need.size(); i++) {
                offset[i] = need[i].min;
            }
            offsetI = offset_box(intersection, offset);
            localI = simplify_box(offsetI, testenv);
            internal_assert(localI[0] == Interval(10, 11));
        }
        {
            BoxIntersection TI(have, need);
            testenv.ref("Rank") = 0;
            testenv.ref("r") = 1;
            // What rank 0 has and rank 1 needs (nothing):
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(11, 9));
            // -4 size is ok: we test for size > 0 to determine empty intersections.
            internal_assert(expr2int(buf.size_of(intersection)) == -4);
        }
    }

    {
        Var y("y");
        DistributedImage<int> in2(10, 20, "in");
        in2.set_domain(x, y);
        in2.placement().distribute(y);
        in2.allocate();
        Func clamped2;
        clamped2(x, y) = in2(clamp(x, 0, in2.global_width() - 1),
                             clamp(y, 0, in2.global_height() - 1));

        Func f("f"), g("g");
        f(x, y) = clamped2(x, y) + clamped2(x, y+1) + 1;
        g(x, y) = f(x, y) + f(x, y+1) + 1;
        f.compute_at(g, y);
        g.compute_root().distribute(y);

        map<string, AbstractBuffer> buffers = func_input_buffers(g);

        internal_assert(buffers.find(f.name()) != buffers.end());
        const AbstractBuffer &buf = buffers.at(f.name());

        Scope<Expr> testenv;
        testenv.push("Rank", Var("r"));
        testenv.push("NumProcessors", numprocs);

        testenv.push(g.name() + ".min.0", 0);
        testenv.push(g.name() + ".max.0", in2.global_width()-1);
        testenv.push(g.name() + ".extent.0", Var(g.name() + ".max.0") - Var(g.name() + ".min.0") + 1);
        testenv.push(g.name() + ".min.1", 0);
        testenv.push(g.name() + ".max.1", in2.global_height()-1);
        testenv.push(g.name() + ".extent.1", Var(g.name() + ".max.1") - Var(g.name() + ".min.1") + 1);


        testenv.push(f.name() + ".min.0", Var(g.name() + ".min.0"));
        testenv.push(f.name() + ".max.0", Var(g.name() + ".max.0"));
        testenv.push(f.name() + ".extent.0", Var(f.name() + ".max.0") - Var(f.name() + ".min.0") + 1);

        testenv.push(f.name() + ".min.1", Var(g.name() + ".min.1"));
        testenv.push(f.name() + ".max.1", Var(g.name() + ".max.1") + 2);
        testenv.push(f.name() + ".extent.1", Var(f.name() + ".max.1") - Var(f.name() + ".min.1") + 1);

        // Test have/need of the input buffer.
        internal_assert(buffers.find(in2.name()) != buffers.end());
        Box need = simplify_box(buffers.at(in2.name()).need(g.name()), testenv);
        testenv.pop("Rank");
        Box have = simplify_box(buffers.at(in2.name()).have(), testenv);
        testenv.push("Rank", 0);

        {
            testenv.ref("Rank") = 0;
            testenv.push("r", 0);
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(0, 9));
            internal_assert(have_concrete[1] == Interval(0, 9));
            internal_assert(need_concrete[0] == Interval(0, 9));
            internal_assert(need_concrete[1] == Interval(0, 11));
        }
        {
            testenv.ref("Rank") = 1;
            testenv.ref("r") = 1;
            Box have_concrete = simplify_box(have, testenv);
            Box need_concrete = simplify_box(need, testenv);
            internal_assert(have_concrete[0] == Interval(0, 9));
            internal_assert(have_concrete[1] == Interval(10, 19));
            internal_assert(need_concrete[0] == Interval(0, 9));
            internal_assert(need_concrete[1] == Interval(10, 19));
        }
        {
            testenv.ref("Rank") = 1;
            testenv.ref("r") = 0;
            BoxIntersection TI(have, need);
            // What rank 1 has and rank 0 needs.
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(0, 9));
            internal_assert(intersection[1] == Interval(10, 11));
            internal_assert(expr2int(buf.size_of(intersection)) == 10*2*4);

            // Local intersection for rank 1
            vector<Expr> offset;
            for (unsigned i = 0; i < have.size(); i++) {
                offset.push_back(have[i].min);
            }
            Box offsetI = offset_box(intersection, offset);
            Box localI = simplify_box(offsetI, testenv);
            internal_assert(localI[0] == Interval(0, 9));
            internal_assert(localI[1] == Interval(0, 1));

            // Local intersection for rank 0
            for (unsigned i = 0; i < need.size(); i++) {
                offset[i] = need[i].min;
            }
            offsetI = offset_box(intersection, offset);
            localI = simplify_box(offsetI, testenv);
            internal_assert(localI[0] == Interval(0, 9));
            internal_assert(localI[1] == Interval(10, 11));
        }
        {
            BoxIntersection TI(have, need);
            testenv.ref("Rank") = 0;
            testenv.ref("r") = 1;
            // What rank 0 has and rank 1 needs (nothing):
            Box intersection = simplify_box(TI.box(), testenv);
            internal_assert(intersection[0] == Interval(0, 9));
            internal_assert(intersection[1] == Interval(10, 9));
            internal_assert(expr2int(buf.size_of(intersection)) == 0);
        }
    }

    std::cout << "Distribute loops internal test passed" << std::endl;
}

}
}
