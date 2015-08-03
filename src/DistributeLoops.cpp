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
        mins << simplify(b[i].min);
        maxs << simplify(b[i].max);
        if (i < b.size() - 1) {
            mins << ", ";
            maxs << ", ";
        }
    }
    mins << ")";
    maxs << ")";
    return mins.str() + " to " + maxs.str();
}

// Simplify the given box, using the given environment of variable
// to value.
Box simplify_box(const Box &b, const Scope<Expr> &env) {
    Box result(b.size());
    for (unsigned i = 0; i < b.size(); i++) {
        Expr min, max;
        for (auto it = env.cbegin(), ite = env.cend(); it != ite; ++it) {
            if (min.defined()) {
                min = Let::make(it.name(), it.value(), min);
                max = Let::make(it.name(), it.value(), max);
            } else {
                min = Let::make(it.name(), it.value(), b[i].min);
                max = Let::make(it.name(), it.value(), b[i].max);
            }
        }
        result[i] = Interval(simplify(min), simplify(max));
    }
    return result;
}

Box box_to_global(const Box &b, Expr offset) {
    Box result(b.size());
    for (unsigned i = 0; i < b.size(); i++) {
        result[i] = Interval(b[i].min + offset, b[i].max + offset);
    }
    return result;
}

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

Box replace(const Box &b, const string &name, Expr value) {
    Box result(b.size());
    for (unsigned i = 0; i < b.size(); i++) {
        ReplaceVariable replace(name, value);
        result[i] = Interval(replace.mutate(b[i].min),
                             replace.mutate(b[i].max));
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
            if (is_positive_const(simplify(b[i].min - a[i].max))) known_empty = true;
            Expr dim_min = simplify(max(a[i].min, b[i].min));
            Expr dim_max = simplify(min(a[i].max, b[i].max));
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
            // If any dimension's min is greater than (or equal to) its max, the
            // intersection is empty.
            Expr e = GE::make(_box[0].min, _box[0].max);
            for (unsigned i = 1; i < _box.size(); i++) {
                e = Or::make(e, GE::make(_box[i].min, _box[i].max));
            }
            return simplify(e);
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

    bool distributed() const {
        // Images and accessor_* functions are never distributed.
        return _btype != Image && !starts_with(_name, "accessor_");
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

    string partitioned_name() const {
        return _name + "_partitioned";
    }

    Expr size_of(const Box &b) const {
        internal_assert(b.size() > 0);
        Expr num_elems = 1;
        for (unsigned i = 0; i < b.size(); i++) {
            num_elems *= b[i].max - b[i].min + 1;
        }
        return simplify(num_elems * elem_size());
    }
private:
    Type _type;
    BufferType _btype;
    string _name;
    int _dimensions;
    vector<Expr> mins;
    vector<Expr> extents;
    vector<Expr> strides;
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

// Build a list of all input and output buffers using a particular
// variable as an index.
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
                    inputs.push_back(AbstractBuffer(call->image.type(), AbstractBuffer::Image, call->image.name()));
                } else {
                    inputs.push_back(AbstractBuffer(call->param.type(), AbstractBuffer::Image, call->param.name()));
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

    void visit(const Provide *provide) {
        GetVariablesInExpr vars(env);
        for (Expr arg : provide->args) {
            arg.accept(&vars);
        }
        if (vars.names.count(name)) {
            internal_assert(provide->values.size() == 1);
            outputs.push_back(AbstractBuffer(provide->values[0].type(), AbstractBuffer::Image, provide->name));
        }
        IRVisitor::visit(provide);
    }
public:
    string name;
    vector<AbstractBuffer> inputs, outputs;
    FindBuffersUsingVariable(string n) : name(n) {}
};

// Return a list of the input buffers used in the given for loop.
map<string, AbstractBuffer> buffers_required(const For *for_loop) {
    FindBuffersUsingVariable find(for_loop->name);
    for_loop->body.accept(&find);
    vector<AbstractBuffer> buffers(find.inputs.begin(), find.inputs.end());
    map<string, AbstractBuffer> result;

    map<string, Box> required = boxes_required(for_loop);
    for (AbstractBuffer buf : buffers) {
        Box b = required[buf.name()];
        Expr stride = 1;
        buf.set_dimensions(b.size());
        for (unsigned i = 0; i < b.size(); i++) {
            Expr extent = b[i].max - b[i].min + 1;
            buf.set_min(i, b[i].min);
            buf.set_extent(i, extent);
            buf.set_stride(i, stride);
            stride *= extent;
        }
        result[buf.name()] = buf;
    }

    return result;
}

// Return a list of the output buffers used in the given for loop.
map<string, AbstractBuffer> buffers_provided(const For *for_loop) {
    FindBuffersUsingVariable find(for_loop->name);
    for_loop->body.accept(&find);
    vector<AbstractBuffer> buffers(find.outputs.begin(), find.outputs.end());
    map<string, AbstractBuffer> result;

    map<string, Box> provided = boxes_provided(for_loop);
    for (AbstractBuffer buf : buffers) {
        Box b = provided[buf.name()];
        Expr stride = 1;
        buf.set_dimensions(b.size());
        for (unsigned i = 0; i < b.size(); i++) {
            Expr extent = b[i].max - b[i].min + 1;
            buf.set_min(i, b[i].min);
            buf.set_extent(i, extent);
            buf.set_stride(i, stride);
            stride *= extent;
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
                newargs.push_back(simplify(call->args[i] - box[i].min));
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
                newargs.push_back(simplify(provide->args[i] - box[i].min));
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
    Expr rowsize = simplify(b[0].max - b[0].min + 1) * buffer.elem_size();
    Expr bufferaddr = address_of(buffer.name(), bufferoffset);
    Expr scratchaddr = address_of(scratch_name, scratchoffset);

    Stmt copyloop;
    switch (cmd) {
    case Pack:
        copyloop = copy_memory(scratchaddr, bufferaddr, rowsize);
        break;
    case Unpack:
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
        Expr extent = simplify(b[i].max - b[i].min + 1);
        body = LetStmt::make(name + ".min." + std::to_string(i), 0, body);
        body = LetStmt::make(name + ".stride." + std::to_string(i), stride, body);
        extents.push_back(extent);
        stride *= extent;
    }
    return Allocate::make(name, type, extents, const_true(), body);
}

typedef enum { Send, Recv } CommunicateCmd;
// Construct a statement to send/recv the required region of 'buffer'
// (specified by box 'b') between rank 0 and each processor.
Stmt communicate_buffer(CommunicateCmd cmd, const AbstractBuffer &buffer, const Box &b) {
    internal_assert(b.size() > 0);
    const string scratch_name = buffer.name() + "_commscratch";
    Stmt copy, commstmt, othercommstmt;
    Expr numbytes = buffer.size_of(b);
    Expr scratch_address = address_of(scratch_name, 0);
    Expr partition_address = address_of(buffer.partitioned_name(), 0);

    switch (cmd) {
    case Send:
        if (b.size() == 1) {
            scratch_address = address_of(buffer, b[0].min);
            commstmt = Evaluate::make(send(scratch_address, numbytes, Var("Rank")));
            othercommstmt = Evaluate::make(recv(partition_address, numbytes, 0));
            copy = copy_memory(partition_address, scratch_address, numbytes);
        } else {
            Stmt pack = pack_region(Pack, scratch_name, buffer, b);
            commstmt = Block::make(pack, Evaluate::make(send(scratch_address, numbytes, Var("Rank"))));
            othercommstmt = Evaluate::make(recv(partition_address, numbytes, 0));
            copy = pack_region(Pack, buffer.partitioned_name(), buffer, b);
        }
        break;
    case Recv:
        if (b.size() == 1) {
            scratch_address = address_of(buffer, b[0].min);
            commstmt = Evaluate::make(recv(scratch_address, numbytes, Var("Rank")));
            othercommstmt = Evaluate::make(send(partition_address, numbytes, 0));
            copy = copy_memory(scratch_address, partition_address, numbytes);
        } else {
            Stmt unpack = pack_region(Unpack, scratch_name, buffer, b);
            commstmt = Block::make(Evaluate::make(recv(scratch_address, numbytes, Var("Rank"))), unpack);
            othercommstmt = Evaluate::make(send(partition_address, numbytes, 0));
            copy = pack_region(Unpack, buffer.partitioned_name(), buffer, b);
        }
        break;
    }
    Stmt commloop = For::make("Rank", 1, num_processors()-1, ForType::Serial, DeviceAPI::Host, commstmt);
    commloop = Block::make(copy, commloop);
    // Rank 0 sends/recvs; all other ranks issue the complement.
    Stmt body = IfThenElse::make(EQ::make(rank(), 0), commloop, othercommstmt);

    // Allocate scratch
    return allocate_scratch(scratch_name, buffer.type(), b, body);
}

// Construct a statement to send the required region of 'buffer'
// (specified by box 'b') from rank 0 to each processor.
Stmt send_input_buffer(const AbstractBuffer &buffer, const Box &b) {
    return communicate_buffer(Send, buffer, b);
}

// Construct a statement to receive the region of 'buffer'
// (specified by box 'b') provided by each processor back to rank
// 0.
Stmt recv_output_buffer(const AbstractBuffer &buffer, const Box &b) {
    return communicate_buffer(Recv, buffer, b);
}

// Construct a receive loop which receives the required region of each
// input buffer from the rank that provides it.
Stmt recv_all_required_regions(const map<string, Box> &required,
                               const map<string, AbstractBuffer> &inputs) {
    Stmt sendstmt;
    for (const auto it : required) {
        const AbstractBuffer &in = inputs.at(it.first);
        const Box &b = it.second;
        if (!in.distributed()) continue;
        if (sendstmt.defined()) {
            sendstmt = Block::make(sendstmt, send_input_buffer(in, b));
        } else {
            sendstmt = send_input_buffer(in, b);
        }
    }
    return sendstmt;
}

// Construct a send loop which sends the provided region of each
// output buffer to each rank that requires it.
Stmt send_all_provided_regions(const map<string, Box> &provided,
                               const map<string, AbstractBuffer> &outputs) {
    Stmt recvstmt;
    for (const auto it : provided) {
        const AbstractBuffer &out = outputs.at(it.first);
        const Box &b = it.second;
        if (!out.distributed()) continue;
        if (recvstmt.defined()) {
            recvstmt = Block::make(recvstmt, recv_output_buffer(out, b));
        } else {
            recvstmt = recv_output_buffer(out, b);
        }
    }
    return recvstmt;
}

// Allocate "partitioned" input and output buffers with the proper
// sizes for each rank. The allocations are valid throughout
// 'body'.
Stmt allocate_partitioned_buffers(const map<string, Box> &regions,
                                  const map<string, AbstractBuffer> &buffers, Stmt body) {
    Stmt allocates;
    for (const auto it : regions) {
        const AbstractBuffer &buf = buffers.at(it.first);
        const Box &b = it.second;
        if (allocates.defined()) {
            allocates = allocate_scratch(buf.partitioned_name(), buf.type(), b, allocates);
        } else {
            allocates = allocate_scratch(buf.partitioned_name(), buf.type(), b, body);
        }
    }
    return allocates;
}

// Return a new loop that has iterations determined by processor
// rank.
Stmt distribute_loop_iterations(const For *for_loop, ForType newtype) {
    Expr r = Var("Rank");
    Var slice_size("SliceSize");
    Expr newmin = for_loop->min + slice_size*r,
        newmax = newmin + slice_size,
        oldmax = for_loop->min + for_loop->extent;
    // Make sure we don't run over old max.
    Expr newextent = min(newmax, oldmax) - newmin;
    // TODO: choose correct loop type here (parallel if original
    // loop was distributed+parallel).
    Stmt newloop = For::make(for_loop->name, simplify(newmin), simplify(newextent),
                             newtype, for_loop->device_api,
                             for_loop->body);
    return newloop;
}
}

class InjectCommunication : public IRMutator {
public:
    const map<string, AbstractBuffer> &inputs, &outputs;
    const map<string, Box> &rank_required, &rank_provided;
    InjectCommunication(const map<string, AbstractBuffer> &in,
                        const map<string, AbstractBuffer> &out,
                        const map<string, Box> &rreq,
                        const map<string, Box> &rprov) :
        inputs(in), outputs(out), rank_required(rreq), rank_provided(rprov) {}

    using IRMutator::visit;

    void visit(const For *for_loop) {
        if (for_loop->for_type != ForType::Distributed) {
            IRMutator::visit(for_loop);
            return;
        }
        // TODO: choose correct loop type here (parallel if original
        // loop was distributed+parallel).
        Stmt newloop = For::make(for_loop->name, for_loop->min, for_loop->extent,
                                 ForType::Serial, for_loop->device_api,
                                 for_loop->body);

        // Get required regions of input buffers in terms of processor
        // rank variable.
        map<string, Box> required, provided;
        required = boxes_required(newloop);
        provided = boxes_provided(newloop);

        // Construct the receive statements to receive required regions for
        // each input buffer from ranks that provide them.
        Stmt recvstmt = recv_all_required_regions(required, inputs);
        // Update the references in the loop to use the "partitioned" input buffers.
        for (const auto it : required) {
            const AbstractBuffer &in = inputs.at(it.first);
            const Box &b = it.second;
            if (!in.distributed()) continue;
            ChangeDistributedLoopBuffers change(in.name(), in.partitioned_name(), b);
            newloop = change.mutate(newloop);
        }

        // Construct send statements to send output buffer regions
        // to ranks that require them.
        Stmt sendstmt = send_all_provided_regions(provided, outputs);
        // Update the references in the loop to use the "partitioned" output buffers.
        for (const auto it : provided) {
            const AbstractBuffer &out = outputs.at(it.first);
            const Box &b = it.second;
            if (!out.distributed()) continue;
            ChangeDistributedLoopBuffers change(out.name(), out.partitioned_name(), b);
            newloop = change.mutate(newloop);
        }

        newloop = sendstmt.defined() ? Block::make(newloop, sendstmt) : newloop;
        newloop = recvstmt.defined() ? Block::make(recvstmt, newloop) : newloop;

        // Construct allocation statements to allcate the partitioned input and output buffers.
        Stmt allocates;
        if (recvstmt.defined()) {
            allocates = allocate_partitioned_buffers(required, inputs, newloop);
        }
        if (sendstmt.defined()) {
            allocates = allocate_partitioned_buffers(provided, outputs,
                                                     allocates.defined() ? allocates : newloop);
        }
        stmt = allocates.defined() ? allocates : newloop;
    }

};

class DistributeLoopsOnly : public IRMutator {
public:
    using IRMutator::visit;
    void visit(const For *for_loop) {
        IRMutator::visit(for_loop);
        if (for_loop->for_type != ForType::Distributed) {
            return;
        }
        // Split original loop into chunks of iterations for each rank.
        stmt = distribute_loop_iterations(for_loop, for_loop->for_type);
    }
};
    
// For each distributed for loop, mutate its bounds to be determined
// by processor rank.
class DistributeLoops : public IRMutator {
public:
    using IRMutator::visit;
    void visit(const For *for_loop) {
        IRMutator::visit(for_loop);
        if (for_loop->for_type != ForType::Distributed) {
            return;
        }
        // Split original loop into chunks of iterations for each rank.
        Stmt newloop = distribute_loop_iterations(for_loop, ForType::Serial);

        // Get required regions of input buffers in terms of processor
        // rank variable.
        map<string, Box> required, provided;
        required = boxes_required(newloop);
        provided = boxes_provided(newloop);

        // Update the buffer references in the loop to use "local" indices.
        // TODO: only do this for distributed buffers.
        for (const auto it : required) {
            const string &name = it.first;
            const Box &b = it.second;
            ChangeDistributedLoopBuffers change(name, name, b);
            newloop = change.mutate(newloop);
        }
        for (const auto it : provided) {
            const string &name = it.first;
            const Box &b = it.second;
            ChangeDistributedLoopBuffers change(name, name, b);
            newloop = change.mutate(newloop);
        }

        newloop = LetStmt::make("Rank", rank(), LetStmt::make("SliceSize", cast(Int(32), ceil(cast(Float(32), for_loop->extent) / num_processors())), newloop));
        stmt = newloop;
    }
};

// Construct a map of all input and output buffers used in a
// pipeline. The results are a map from buffer name -> AbstractBuffer
// with information about the buffer.
class GetPipelineInputsAndOutputs : public IRVisitor {
    Scope<Expr> env;
public:
    // Maps from buffer name -> region used expressed in terms of
    // processor rank.
    map<string, Box> rank_required, rank_provided;
    map<string, AbstractBuffer> inputs, outputs;

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

    void visit(const For *for_loop) {
        map<string, AbstractBuffer> in, out;
        in = buffers_required(for_loop);
        out = buffers_provided(for_loop);
        inputs.insert(in.begin(), in.end());
        outputs.insert(out.begin(), out.end());

        map<string, Box> required, provided;

        required = boxes_required(for_loop);
        provided = boxes_provided(for_loop);

        // TODO: if number of processors doesn't divide the global
        // extent evenly, this won't work, because some processor will
        // have a different extent.
        Expr offset = for_loop->for_type == ForType::Distributed ?
            for_loop->extent * Var("Rank") : 0;

        // Convert required/provided regions of input/output buffers
        // to be in terms of processor rank variable.
        for (const auto it : required) {
            Box b = box_to_global(simplify_box(it.second, env), offset);
            rank_required[it.first] = b;
        }
        for (const auto it : provided) {
            Box b = box_to_global(simplify_box(it.second, env), offset);
            rank_provided[it.first] = b;
        }

        IRVisitor::visit(for_loop);
    }

    // For all outputs that have an input of the same name, set the
    // output type to the input type. This is required because we're
    // unable to tell whether a Provide node is for an Image output
    // buffer or not, so we must correct the buffer type here
    // accordingly.
    void settypes() {
        for (auto &it : outputs) {
            auto in = inputs.find(it.first);
            if (in != inputs.end()) {
                it.second.set_buffer_type(in->second.buffer_type());
            }
        }
    }
};

Stmt distribute_loops_only(Stmt s) {
    return DistributeLoopsOnly().mutate(s);
}

Stmt distribute_loops(Stmt s) {
    GetPipelineInputsAndOutputs getio;
    s.accept(&getio);
    getio.settypes();
    s = DistributeLoops().mutate(s);

    // s = InjectCommunication(getio.inputs, getio.outputs,
    //                         getio.rank_required, getio.rank_provided).mutate(s);
    return s;
}

}
}
