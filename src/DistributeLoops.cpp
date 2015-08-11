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
        result[i] = Interval(min, max);
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
    typedef enum { Halide, Image } BufferType;

    AbstractBuffer() : _dimensions(-1) {}
    AbstractBuffer(Type type, BufferType btype, const string &name) :
        _type(type), _btype(btype), _name(name), _dimensions(-1) {}
    AbstractBuffer(Type type, BufferType btype, const string &name, const Buffer &buffer) :
        _type(type), _btype(btype), _name(name), _dimensions(-1) {
        internal_assert(btype == Image);
        internal_assert(buffer.defined());
        internal_assert(buffer.distributed());
        Expr stride = 1;
        for (int i = 0; i < buffer.dimensions(); i++) {
            Expr min = buffer.local_min(i);
            Expr max = min + buffer.local_extent(i) - 1;
            Expr extent = max - min + 1;
            strides.push_back(stride);
            stride *= extent;
            _bounds.push_back(Interval(min, max));
        }
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

    Expr stride(int dim) const {
        internal_assert(dim >= 0 && dim < (int)strides.size());
        return strides[dim];
    }

    void set_stride(int dim, Expr stride) {
        if (dim >= (int)strides.size()) {
            strides.resize(dim+1);
        }
        strides[dim] = stride;
    }

    Type type() const {
        return _type;
    }

    BufferType buffer_type() const {
        return _btype;
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
        return _name + "_extended";
    }

    Expr size_of(const Box &b) const {
        internal_assert(b.size() > 0);
        Expr num_elems = 1;
        for (unsigned i = 0; i < b.size(); i++) {
            num_elems *= b[i].max - b[i].min + 1;
        }
        return num_elems * elem_size();
    }

    const Box &bounds() const {
        internal_assert(!_bounds.empty()) << _name;
        return _bounds;
    }

    void set_bounds(const Box &b) {
        internal_assert(_bounds.empty());
        set_dimensions(b.size());
        _bounds = Box(b.size());
        Expr stride = 1;
        for (unsigned i = 0; i < b.size(); i++) {
            Expr extent = b[i].max - b[i].min + 1;
            strides.push_back(stride);
            stride *= extent;
            _bounds[i] = Interval(b[i].min, b[i].max);
        }
    }
private:
    Type _type;
    BufferType _btype;
    string _name;
    int _dimensions;
    vector<Expr> mins;
    vector<Expr> extents;
    vector<Expr> strides;
    Box _bounds;
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
            if (call->call_type == Call::Image) {
                if (call->image.defined()) {
                    inputs.push_back(AbstractBuffer(call->image.type(), AbstractBuffer::Image, call->image.name(), (Buffer)call->image));
                } else {
                    inputs.push_back(AbstractBuffer(call->param.type(), AbstractBuffer::Image, call->param.name(), call->param.get_buffer()));
                }
            } else if (call->call_type == Call::Halide) {
                internal_assert(call->func.outputs() == 1);
                inputs.push_back(AbstractBuffer(call->func.output_types()[0], AbstractBuffer::Halide, call->func.name()));
            } else {
                internal_assert(false) << "Unhandled call type.\n";
            }

        }
        IRVisitor::visit(call);
    }
public:
    string name;
    vector<AbstractBuffer> inputs;
    FindBuffersUsingVariable(string n) : name(n) {}
};

// Return a list of the input buffers used in the given for loop.
map<string, AbstractBuffer> buffers_required(const For *for_loop) {
    FindBuffersUsingVariable find(for_loop->name);
    for_loop->body.accept(&find);
    vector<AbstractBuffer> buffers(find.inputs.begin(), find.inputs.end());
    map<string, AbstractBuffer> result;

    for (AbstractBuffer buf : buffers) {
        if (buf.buffer_type() == AbstractBuffer::Image) {
            internal_assert(!buf.bounds().empty());
        }
        result[buf.name()] = buf;
    }

    return result;
}

// Return total number of processors available.
Expr num_processors() {
    return Call::make(Int(32), "halide_do_distr_size", {}, Call::Extern);
}

// Return rank of the current processor.
Expr rank() {
    return Call::make(Int(32), "halide_do_distr_rank", {}, Call::Extern);
}

// Insert call to send 'count' bytes starting at 'address' to 'rank'.
Expr send(Expr address, Expr count, Expr rank) {
    return Call::make(Int(32), "halide_do_distr_send", {address, count, rank}, Call::Extern);
}

// Insert call to receive 'count' bytes from 'rank' to buffer starting at 'address'.
Expr recv(Expr address, Expr count, Expr rank) {
    return Call::make(Int(32), "halide_do_distr_recv", {address, count, rank}, Call::Extern);
}

// Return the (symbolic) address of the given buffer at the given
// byte index.
Expr address_of(const string &buffer, Expr index) {
    Expr first_elem = Load::make(UInt(8), buffer, index, Buffer(), Parameter());
    return Call::make(Handle(), Call::address_of, {first_elem}, Call::Intrinsic);
}

// Return the (symbolic) address of the given buffer at the given
// element index.
Expr address_of(const AbstractBuffer &buffer, Expr index) {
    // A load of UInt(8) will take an index in bytes; we are given
    // an index in elements.
    index *= buffer.elem_size();
    return address_of(buffer.name(), index);
}

// Construct a statement to copy 'size' bytes from src to dest.
Stmt copy_memory(Expr dest, Expr src, Expr size) {
    return Evaluate::make(Call::make(UInt(8), Call::copy_memory,
                                    {dest, src, size}, Call::Intrinsic));
}

class ChangeDistributedLoopBuffers : public IRMutator {
    string name, newname;
    const Box &box;
public:
    using IRMutator::visit;
    ChangeDistributedLoopBuffers(const string &n, const string &nn, const Box &b) :
        name(n), newname(nn), box(b) {}

    void visit(const Call *call) {
        if (call->name == name) {
            vector<Expr> newargs;
            for (unsigned i = 0; i < box.size(); i++) {
                newargs.push_back(call->args[i] - box[i].min);
            }
            expr = Call::make(call->type, newname, newargs, call->call_type,
                              call->func, call->value_index, call->image,
                              call->param);
        } else {
            IRMutator::visit(call);
        }
    }

    void visit(const Provide *provide) {
        if (provide->name == name) {
            vector<Expr> newargs;
            for (unsigned i = 0; i < box.size(); i++) {
                newargs.push_back(provide->args[i] - box[i].min);
            }
            stmt = Provide::make(newname, provide->values, newargs);
        } else {
            IRMutator::visit(provide);
        }
    }
};

typedef enum { Pack, Unpack } PackCmd;
// Construct a statement to pack/unpack the given box of the given buffer
// to/from the contiguous scratch region of memory with the given name.
Stmt pack_region(PackCmd cmd, const string &scratch_name, const AbstractBuffer &buffer, const Box &b) {
    internal_assert(b.size() > 0);
    vector<Var> dims;
    for (unsigned i = 0; i < b.size(); i++) {
        dims.push_back(Var(buffer.name() + "_dim" + std::to_string(i)));
    }

    // Construct src/dest pointer expressions as expressions in
    // terms of the box dimension variables.
    Expr bufferoffset = 0, scratchoffset = 0;
    Expr scratchstride = b[0].max - b[0].min + 1;
    for (unsigned i = 1; i < b.size(); i++) {
        Expr extent = b[i].max - b[i].min + 1;
        Expr dim = dims[i];
        bufferoffset += (dim + b[i].min) * buffer.stride(i) * buffer.elem_size();
        scratchoffset += dim * scratchstride * buffer.elem_size();
        scratchstride *= extent;
    }

    // Construct loop nest to copy each contiguous row.  TODO:
    // ensure this nesting is in the correct row/column major
    // order.
    Expr rowsize = (b[0].max - b[0].min + 1) * buffer.elem_size();
    Expr bufferaddr;
    Expr scratchaddr = address_of(scratch_name, scratchoffset);

    Stmt copyloop;
    switch (cmd) {
    case Pack:
        bufferaddr = address_of(buffer.name(), bufferoffset);
        copyloop = copy_memory(scratchaddr, bufferaddr, rowsize);
        break;
    case Unpack:
        bufferaddr = address_of(buffer.extended_name(), bufferoffset);
        copyloop = copy_memory(bufferaddr, scratchaddr, rowsize);
        break;
    }
    for (int i = b.size() - 1; i >= 1; i--) {
        copyloop = For::make(dims[i].name(), 0, b[i].max - b[i].min + 1,
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
Stmt copy_on_node_data(const map<string, Box> &required,
                       const map<string, AbstractBuffer> &inputs) {
    Stmt copy;
    for (const auto it : required) {
        const string &name = it.first;
        const AbstractBuffer &in = inputs.at(name);
        const Box &have = in.bounds();

        // TODO: may have to copy to destination offset other than 0
        // TODO: may have to copy multidim buffers by contiguous
        // section, as the destination buffer may have a different
        // shape (strides).

        //internal_assert(have.size() == 1);
        Expr dest = address_of(in.extended_name(), 0), src = address_of(in.name(), 0);
        Expr numbytes = in.size_of(have);
        if (copy.defined()) {
            copy = Block::make(copy, copy_memory(dest, src, numbytes));
        } else {
            copy = copy_memory(dest, src, numbytes);
        }
    }
    return copy;
}

typedef enum { Send, Recv } CommunicateCmd;
// Generate communication code to send/recv the intersection of the
// 'have' and 'need' regions of 'buf'.
Stmt communicate_intersection(CommunicateCmd cmd, const AbstractBuffer &buf, const Box &have, const Box &need) {
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
    const string scratch_name = buf.name() + "_commscratch";

    // Convert the intersection box to "local" coordinates (the
    // extended buffer counts from 0). This just means subtracting the
    // min global coordinate from the intersection bounds (which are
    // also global) to get a local coordinate starting from 0.
    vector<Expr> offset_have, offset_need;
    internal_assert(have.size() == need.size());
    for (unsigned i = 0; i < have.size(); i++) {
        offset_have.push_back(have[i].min);
        offset_need.push_back(need[i].min);
    }
    Box local_have = offset_box(I.box(), offset_have),
        local_need = offset_box(I.box(), offset_need);

    switch (cmd) {
    case Send:
        if (local_have.size() == 1) {
            addr = address_of(buf.name(), local_have[0].min * buf.elem_size());
            commstmt = IfThenElse::make(cond, Evaluate::make(send(addr, numbytes, Var("r"))));
        } else {
            Stmt pack = pack_region(Pack, scratch_name, buf, local_have);
            addr = address_of(scratch_name, 0);
            commstmt = IfThenElse::make(cond, Block::make(pack, Evaluate::make(send(addr, numbytes, Var("r")))));
        }
        break;
    case Recv:
        if (local_need.size() == 1) {
            addr = address_of(buf.extended_name(), local_need[0].min * buf.elem_size());
            commstmt = IfThenElse::make(cond, Evaluate::make(recv(addr, numbytes, Var("r"))));
        } else {
            Stmt unpack = pack_region(Unpack, scratch_name, buf, local_need);
            addr = address_of(scratch_name, 0);
            commstmt = IfThenElse::make(cond, Block::make(Evaluate::make(recv(addr, numbytes, Var("r"))), unpack));
        }
        break;
    }
    // TODO: we have to allocate the communication buffer inside the
    // loop because the size of the intersection depends on "r". Can
    // we do something smarter?
    commstmt = LetStmt::make("msgsize", buf.size_of(I.box()), commstmt);
    commstmt = allocate_scratch(scratch_name, buf.type(), I.box(), commstmt);
    commstmt = For::make("r", 0, Var("NumProcessors"), ForType::Serial, DeviceAPI::Host, commstmt);
    return commstmt;
}

// For each required region, generate communication code between ranks
// that own data needed by other ranks.
Stmt exchange_data(const map<string, Box> &required,
                   const map<string, AbstractBuffer> &inputs) {
    Stmt sendloop, recvloop;
    for (const auto it : required) {
        const string &name = it.first;
        const Box &need = it.second;
        const AbstractBuffer &in = inputs.at(name);
        const Box &have = in.bounds();

        if (sendloop.defined()) {
            sendloop = Block::make(sendloop, communicate_intersection(Send, in, have, need));
        } else {
            sendloop = communicate_intersection(Send, in, have, need);
        }

        if (recvloop.defined()) {
            recvloop = Block::make(recvloop, communicate_intersection(Recv, in, have, need));
        } else {
            recvloop = communicate_intersection(Recv, in, have, need);
        }
    }
    internal_assert(sendloop.defined() == recvloop.defined());
    if (!sendloop.defined()) {
        return Stmt();
    } else {
        return Block::make(sendloop, recvloop);
    }
}

// Change all uses of the original input buffers to use the extended
// buffers, and modify output buffer indices to be "local" indices
// (instead of global).
Stmt update_io_buffers(Stmt loop, const map<string, Box> &required,
                       const map<string, AbstractBuffer> &inputs,
                       const map<string, Box> &provided) {
    for (const auto it : required) {
        const AbstractBuffer &in = inputs.at(it.first);
        const Box &b = it.second;
        ChangeDistributedLoopBuffers change(in.name(), in.extended_name(), b);
        loop = change.mutate(loop);
    }

    for (const auto it : provided) {
        const string &name = it.first;
        const Box &b = it.second;
        ChangeDistributedLoopBuffers change(name, name, b);
        loop = change.mutate(loop);
    }
    return loop;
}

// Allocate extended buffers for the given body.
Stmt allocate_extended_buffers(Stmt body, const map<string, Box> &required,
                               const map<string, AbstractBuffer> &inputs) {
    Stmt allocates = body;
    for (const auto it : required) {
        const AbstractBuffer &in = inputs.at(it.first);
        const Box &b = it.second;
        allocates = allocate_scratch(in.extended_name(), in.type(), b, allocates);
    }
    return allocates;
}

}

class InjectCommunication : public IRMutator {
public:
    Scope<Expr> env;

    const map<string, AbstractBuffer> &inputs;
    InjectCommunication(const map<string, AbstractBuffer> &in) : inputs(in) {}

    using IRMutator::visit;

    void visit(const LetStmt *let) {
        env.push(let->name, let->value);
        IRMutator::visit(let);
        env.pop(let->name);
    }

    void visit(const Let *let) {
        env.push(let->name, let->value);
        IRMutator::visit(let);
        env.pop(let->name);
    }

    void visit(const For *for_loop) {
        // if (for_loop->for_type != ForType::Distributed) {
        //     IRMutator::visit(for_loop);
        //     return;
        // }

        map<string, Box> required, provided;
        required = boxes_required(for_loop);
        provided = boxes_provided(for_loop);

        // Boxes are initially in terms of loop_min/max/extent. We
        // need them in terms of processor rank, which we can
        // accomplish by simplifying them with the current environment
        // (which has the loop_* let statements).
        for (auto it : required) {
            required[it.first] = simplify_box(it.second, env);
        }

        for (auto it : provided) {
            provided[it.first] = simplify_box(it.second, env);
        }

        Stmt newloop = for_loop;

        // For each Image input buffer:
        // Allocate scratch buffer big enough for what I have + ghost zone
        // Copy my stuff into scratch
        // Send/recv from somebody else using rank_required/provided.
        // Replace input buffer refs with scratch

        Stmt copy = copy_on_node_data(required, inputs);
        if (copy.defined()) {
            newloop = Block::make(copy, newloop);
        }

        Stmt border_exchange = exchange_data(required, inputs);
        if (border_exchange.defined()) {
            newloop = Block::make(border_exchange, newloop);
        }

        newloop = update_io_buffers(newloop, required, inputs, provided);
        newloop = allocate_extended_buffers(newloop, required, inputs);

        stmt = newloop;
    }
};

// For each distributed for loop, mutate its bounds to be determined
// by processor rank.
class DistributeLoops : public IRMutator {
public:
    set<string> slice_size_inserted;
    const map<string, Expr> &distributed_bounds;
    DistributeLoops(const map<string, Expr> &bounds) : distributed_bounds(bounds) {}

    using IRMutator::visit;
    void visit(const LetStmt *let) {
        if (distributed_bounds.find(let->name) != distributed_bounds.end()) {
            string loop_var = remove_suffix(let->name);
            Expr oldmin = distributed_bounds.at(loop_var + ".loop_min"),
                oldmax = distributed_bounds.at(loop_var + ".loop_max"),
                oldextent = distributed_bounds.at(loop_var + ".loop_extent");
            Expr slice_size = cast(Int(32), ceil(cast(Float(32), oldextent) / Var("NumProcessors")));
            Expr newmin = oldmin + Var(loop_var + ".SliceSize") * Var("Rank"),
                newmax = newmin + Var(loop_var + ".SliceSize") - 1;
            // Make sure we don't run over old max.
            Expr newextent = min(newmax, oldmax) - newmin + 1;
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
            }
        } else {
            IRMutator::visit(let);
        }
    }

    void visit(const For *for_loop) {
        IRMutator::visit(for_loop);
        if (for_loop->for_type == ForType::Distributed) {
            stmt = For::make(for_loop->name, for_loop->min, for_loop->extent,
                             ForType::Serial, for_loop->device_api,
                             for_loop->body);
        }
    }
private:
    // Removes last token of the string, delimited by '.'
    string remove_suffix(const string &str) const {
        size_t lastdot = str.find_last_of(".");
        if (lastdot != std::string::npos) {
            return str.substr(0, lastdot);
        } else {
            return str;
        }
    }
};

class FindDistributedLoops : public IRVisitor {
public:
    map<string, Expr> distributed_bounds;

    using IRVisitor::visit;

    void visit(const LetStmt *let) {
        env.push(let->name, let->value);
        IRVisitor::visit(let);
        env.pop(let->name);
    }

    void visit(const For *for_loop) {
        if (for_loop->for_type == ForType::Distributed) {
            for (auto it = env.begin(), ite = env.end(); it != ite; ++it) {
                if (starts_with(it.name(), for_loop->name)) {
                    distributed_bounds[it.name()] = it.value();
                }
            }
        }
        IRVisitor::visit(for_loop);
    }
private:
    Scope<Expr> env;
};

// Construct a map of all input buffers used in a pipeline. The
// results are a map from buffer name -> AbstractBuffer with
// information about the buffer.
class GetPipelineInputs : public IRVisitor {
public:
    map<string, AbstractBuffer> inputs;

    using IRVisitor::visit;

    void visit(const For *for_loop) {
        map<string, AbstractBuffer> in;
        in = buffers_required(for_loop);
        inputs.insert(in.begin(), in.end());
        IRVisitor::visit(for_loop);
    }
};

// Set the bounds for all non-Image input buffers based on the region
// provided by their producing loops. This should take place *after*
// loops have been distributed, otherwise the bounds set will be
// global values.
class SetInputBufferBounds : public IRVisitor {
public:
    Scope<Expr> env;
    map<string, AbstractBuffer> &inputs;
    SetInputBufferBounds(map<string, AbstractBuffer> &in) : inputs(in) {}

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

    void visit(const For *for_loop) {
        map<string, Box> provided = boxes_provided(for_loop);
        for (auto it : provided) {
            if (inputs.find(it.first) != inputs.end()) {
                AbstractBuffer &buf = inputs.at(it.first);
                internal_assert(buf.buffer_type() != AbstractBuffer::Image);
                buf.set_bounds(simplify_box(it.second, env));
            }
        }
        IRVisitor::visit(for_loop);
    }
};

Stmt distribute_loops_only(Stmt s) {
    FindDistributedLoops find;
    s.accept(&find);
    return DistributeLoops(find.distributed_bounds).mutate(s);
}

Stmt distribute_loops(Stmt s) {
    FindDistributedLoops find;
    s.accept(&find);
    s = DistributeLoops(find.distributed_bounds).mutate(s);
    GetPipelineInputs getio;
    s.accept(&getio);
    SetInputBufferBounds setb(getio.inputs);
    s.accept(&setb);
    s = InjectCommunication(getio.inputs).mutate(s);
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

    FindDistributedLoops find;
    s.accept(&find);
    s = DistributeLoops(find.distributed_bounds).mutate(s);
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
    Stmt s = partial_lower(f);
    GetPipelineInputs getio;
    s.accept(&getio);
    SetInputBufferBounds setb(getio.inputs);
    s.accept(&setb);
    return getio.inputs;
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

        map<string, Box> boxes_provided = func_boxes_provided(f),
            boxes_required = func_boxes_required(f);
        map<string, AbstractBuffer> inputs = func_input_buffers(f);

        const AbstractBuffer &buf = inputs.at(in.name());
        const Box &have = buf.bounds();
        const Box &need = boxes_required.at(in.name());

        Scope<Expr> testenv;
        testenv.push("Rank", 0);
        testenv.push("NumProcessors", numprocs);
        testenv.push(f.name() + ".s0.x.min", 0);
        testenv.push(f.name() + ".s0.x.max", w-1);

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
        map<string, Box> boxes_provided = func_boxes_provided(f),
            boxes_required = func_boxes_required(f);
        map<string, AbstractBuffer> inputs = func_input_buffers(f);

        const AbstractBuffer &buf = inputs.at(in.name());
        const Box &b = buf.bounds();
        const Box &req = boxes_required.at(in.name());

        Scope<Expr> testenv;
        testenv.push("Rank", Var("r"));
        testenv.push("NumProcessors", numprocs);
        testenv.push(f.name() + ".s0.x.min", 0);
        testenv.push(f.name() + ".s0.x.max", w-1);

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

        map<string, Box> boxes_provided = func_boxes_provided(g),
            boxes_required = func_boxes_required(g);
        map<string, AbstractBuffer> inputs = func_input_buffers(g);

        const AbstractBuffer &buf = inputs.at(f.name());
        const Box &b = buf.bounds();
        const Box &req = boxes_required.at(f.name());


        Scope<Expr> testenv;
        testenv.push("Rank", Var("r"));
        testenv.push("NumProcessors", numprocs);
        testenv.push(g.name() + ".s0.x.min", 0);
        testenv.push(g.name() + ".s0.x.max", w-1);
        testenv.push(f.name() + ".s0.x.min", Var(g.name() + ".s0.x.min"));
        testenv.push(f.name() + ".s0.x.max", Var(g.name() + ".s0.x.max") + 1);

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
            internal_assert(have_concrete[0] == Interval(11, 20));
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
    }

    std::cout << "Distribute loops internal test passed" << std::endl;
}

}
}
