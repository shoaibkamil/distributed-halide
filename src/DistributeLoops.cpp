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

// Helper class that wraps information common to Buffer and Parameter classes.
// Also provides a wrapper for Provide nodes which do not have buffer references.
class AbstractBuffer {
public:
    AbstractBuffer(const Buffer &buffer) {
        _type = buffer.type();
        _name = buffer.name();
        _dimensions = buffer.dimensions();
        for (int d = 0; d < buffer.dimensions(); d++) {
            mins.push_back(buffer.min(d));
            extents.push_back(buffer.extent(d));
            //strides.push_back(buffer.stride(d));
        }
    }

    AbstractBuffer(const Parameter &param) {
        _type = param.type();
        _name = param.name();
        _dimensions = param.dimensions();
        for (int d = 0; d < param.dimensions(); d++) {
            mins.push_back(param.min_constraint(d));
            extents.push_back(param.extent_constraint(d));
            //strides.push_back(param.stride_constraint(d));
        }
    }

    AbstractBuffer(const Provide *provide) {
        _type = provide->values[0].type();
        _name = provide->name;
        _dimensions = -1;
    }

    // int dimensions() const {
    //     internal_assert(_dimensions >= 0) << "Called dimensions on AbstractBuffer of Provide type.\n";
    //     return _dimensions;
    // }

    // Expr extent(int dim) const {
    //     internal_assert(dim >= 0 && dim < (int)extents.size());
    //     return extents[dim];
    // }

    // Expr min(int dim) const {
    //     internal_assert(dim >= 0 && dim < (int)mins.size());
    //     return mins[dim];
    // }

    // Expr stride(int dim) const {
    //     internal_assert(dim >= 0 && dim < (int)strides.size());
    //     return strides[dim];
    // }

    Type type() const {
        return _type;
    }

    const string &name() const {
        return _name;
    }

    string partitioned_name() const {
        return _name + "_partitioned";
    }

    Expr elem_size() const {
        return _type.bytes();
    }

    Expr size_of(const Box &b) const {
        Expr num_elems = 1;
        for (unsigned i = 0; i < b.size(); i++) {
            num_elems *= b[i].max - b[i].min + 1;
        }
        return simplify(num_elems * elem_size());
    }
private:
    string _name;
    Type _type;
    int _dimensions;
    vector<Expr> mins;
    vector<Expr> extents;
    //    vector<Expr> strides;
};
}

class DistributeLoops : public IRMutator {
    // Build a set of all the variables referenced.
    class GetVariablesInExpr : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Variable *var) {
            names.insert(var->name);
            IRVisitor::visit(var);
        }
    public:
        set<string> names;
    };

    // Build a list of all input and output buffers using a particular
    // variable as an index.
    class FindBuffersUsingVariable : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Call *call) {
            GetVariablesInExpr vars;
            for (Expr arg : call->args) {
                arg.accept(&vars);
            }
            if (vars.names.count(name)) {
                internal_assert(call->call_type == Call::Image);
                if (call->image.defined()) {
                    inputs.push_back(AbstractBuffer(call->image));
                } else {
                    inputs.push_back(AbstractBuffer(call->param));
                }

            }
            IRVisitor::visit(call);
        }

        void visit(const Provide *provide) {
            GetVariablesInExpr vars;
            for (Expr arg : provide->args) {
                arg.accept(&vars);
            }
            if (vars.names.count(name)) {
                internal_assert(provide->values.size() == 1);
                outputs.push_back(AbstractBuffer(provide));
            }
            IRVisitor::visit(provide);
        }

        map<string, Expr> elem_sizes;
    public:
        string name;
        vector<AbstractBuffer> inputs, outputs;
        FindBuffersUsingVariable(string n) : name(n) {}
    };


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

    // Return total number of processors available.
    Expr num_processors() const {
        return Call::make(Int(32), "halide_do_distr_size", {}, Call::Extern);
    }

    // Return rank of the current processor.
    Expr rank() const {
        return Call::make(Int(32), "halide_do_distr_rank", {}, Call::Extern);
    }

    // Insert call to send 'count' bytes starting at 'address' to 'rank'.
    Expr send(Expr address, Expr count, Expr rank) const {
        return Call::make(Int(32), "halide_do_distr_send", {address, count, rank}, Call::Extern);
    }

    // Insert call to receive 'count' bytes from 'rank' to buffer starting at 'address'.
    Expr recv(Expr address, Expr count, Expr rank) const {
        return Call::make(Int(32), "halide_do_distr_recv", {address, count, rank}, Call::Extern);
    }

    // Return the (symbolic) address of the given buffer at the given
    // byte index.
    Expr address_of(const string &buffer, Expr index) const {
        Expr first_elem = Load::make(UInt(8), buffer, index, Buffer(), Parameter());
        return Call::make(Handle(), Call::address_of, {first_elem}, Call::Intrinsic);
    }

    // Return the (symbolic) address of the given buffer at the given
    // element index.
    Expr address_of(const AbstractBuffer &buffer, Expr index) const {
        // A load of UInt(8) will take an index in bytes; we are given
        // an index in elements.
        index *= buffer.elem_size();
        return address_of(buffer.name(), index);
    }

    // Return the (symbolic) address of the given n-D buffer at the
    // given element index.
    // Expr address_of(const AbstractBuffer &buffer, const vector<Expr> &index) const {
    //     Expr idx = 0;
    //     for (int i = 0; i < (int)index.size(); i++) {
    //         idx += i*buffer.extent(i) + index[i];
    //     }
    //     return address_of(buffer.name(), idx);
    // }

    // Construct a statement to copy 'size' bytes from src to dest.
    Stmt copy_memory(Expr dest, Expr src, Expr size) const {
        return Evaluate::make(Call::make(UInt(8), Call::copy_memory,
                                         {dest, src, size}, Call::Intrinsic));
    }

    // Construct a statement to pack the given box of the given buffer
    // into the contiguous region of memory with the given name.
    Stmt pack_region(const string &dest_name, const AbstractBuffer &buffer, const Box &b) const {
        vector<Var> dims;
        for (unsigned i = 0; i < b.size(); i++) {
            dims.push_back(Var(buffer.name() + "_dim" + std::to_string(i)));
        }

        // Construct src/dest pointer expressions as expressions in
        // terms of the box dimension variables.
        Expr srcoffset = 0, destoffset = 0;
        Expr stride = b[0].max - b[0].min + 1;
        for (unsigned i = 1; i < b.size(); i++) {
            Expr extent = b[i].max - b[i].min + 1;
            Expr dim = dims[i];
            srcoffset += (dim + b[i].min) * stride * buffer.elem_size();
            destoffset += dim * stride * buffer.elem_size();
            stride *= extent;
        }

        // Construct loop nest to copy each contiguous row.  TODO:
        // ensure this nesting is in the correct row/column major
        // order.
        Expr rowsize = simplify(b[0].max - b[0].min + 1) * buffer.elem_size();
        Expr src = address_of(buffer.name(), srcoffset);
        Expr dest = address_of(dest_name, destoffset);
        Stmt copyloop = copy_memory(dest, src, rowsize);
        for (int i = b.size() - 1; i >= 1; i--) {
            copyloop = For::make(dims[i].name(), 0, b[i].max - b[i].min + 1,
                                 ForType::Serial, DeviceAPI::Host, copyloop);
        }
        return copyloop;
    }

    // Construct a statement to unpack the given contiguous region of
    // memory with the given name into the given box of the given
    // buffer.
    Stmt unpack_region(const string &src_name, const AbstractBuffer &buffer, const Box &b) const {
        vector<Var> dims;
        for (unsigned i = 0; i < b.size(); i++) {
            dims.push_back(Var(buffer.name() + "_dim" + std::to_string(i)));
        }

        // Construct src/dest pointer expressions as expressions in
        // terms of the box dimension variables.
        Expr srcoffset = 0, destoffset = 0;
        Expr stride = b[0].max - b[0].min + 1;
        for (unsigned i = 1; i < b.size(); i++) {
            Expr extent = b[i].max - b[i].min + 1;
            Expr dim = dims[i];
            srcoffset += dim * stride * buffer.elem_size();
            destoffset += (dim + b[i].min) * stride * buffer.elem_size();
            stride *= extent;
        }

        // Construct loop nest to copy each contiguous row.  TODO:
        // ensure this nesting is in the correct row/column major
        // order.
        Expr rowsize = simplify(b[0].max - b[0].min + 1) * buffer.elem_size();
        Expr src = address_of(src_name, srcoffset);
        Expr dest = address_of(buffer.name(), destoffset);
        Stmt copyloop = copy_memory(dest, src, rowsize);
        for (int i = b.size() - 1; i >= 1; i--) {
            copyloop = For::make(dims[i].name(), 0, b[i].max - b[i].min + 1,
                                 ForType::Serial, DeviceAPI::Host, copyloop);
        }
        return copyloop;
    }

    // Return a new loop that has iterations determined by processor
    // rank.
    Stmt distribute_loop_iterations(const For *for_loop) const {
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
                                 ForType::Serial, for_loop->device_api,
                                 for_loop->body);
        return newloop;
    }

    typedef enum { Send, Recv } CommunicateCmd;
    // Construct a statement to send/recv the required region of 'buffer'
    // (specified by box 'b') between rank 0 and each processor.
    Stmt communicate_buffer(CommunicateCmd cmd, const AbstractBuffer &buffer, const Box &b) const {
        internal_assert(b.size() > 0);
        switch (b.size()) {
        case 1: {
            Expr rowsize = b[0].max - b[0].min + 1;
            Expr rowbytes = rowsize * buffer.elem_size();

            // Loop through each rank (not including 0) to evaluate
            // the box needed (since the box given is expressed in
            // terms of a rank variable), then communicate it.
            Expr address = address_of(buffer, b[0].min);
            Expr partitioned_addr = address_of(buffer.partitioned_name(), 0);
            Stmt commstmt, othercommstmt, copy;
            switch (cmd) {
            case Send:
                commstmt = Evaluate::make(send(address, rowbytes, Var("Rank")));
                othercommstmt = Evaluate::make(recv(partitioned_addr, rowbytes, 0));
                copy = copy_memory(partitioned_addr, address, rowbytes);
                break;
            case Recv:
                commstmt = Evaluate::make(recv(address, rowbytes, Var("Rank")));
                othercommstmt = Evaluate::make(send(partitioned_addr, rowbytes, 0));
                copy = copy_memory(address, partitioned_addr, rowbytes);
                break;
            }
            Stmt commloop = For::make("Rank", 1, num_processors()-1, ForType::Serial, DeviceAPI::Host, commstmt);

            commloop = Block::make(copy, commloop);

            // Rank 0 sends/recvs; all other ranks issue the complement.
            return IfThenElse::make(EQ::make(rank(), 0), commloop, othercommstmt);
        }
        default:
            string scratch_name = buffer.name() + "_commscratch";

            Expr numbytes = buffer.size_of(b);
            Expr address = address_of(scratch_name, 0);
            Expr partitioned_addr = address_of(buffer.partitioned_name(), 0);
            Stmt commstmt, othercommstmt, copy, packunpack;
            switch (cmd) {
            case Send:
                // Pack buffer into scratch
                packunpack = pack_region(scratch_name, buffer, b);
                commstmt = Block::make(packunpack, Evaluate::make(send(address, numbytes, Var("Rank"))));
                othercommstmt = Evaluate::make(recv(partitioned_addr, numbytes, 0));
                copy = Block::make(packunpack, copy_memory(partitioned_addr, address, numbytes));
                break;
            case Recv:
                // Unpack buffer from scratch
                packunpack = unpack_region(scratch_name, buffer, b);
                commstmt = Block::make(Evaluate::make(recv(address, numbytes, Var("Rank"))),
                                       packunpack);
                othercommstmt = Evaluate::make(send(partitioned_addr, numbytes, 0));
                copy = Block::make(copy_memory(address, partitioned_addr, numbytes), packunpack);
                break;
            }
            Stmt commloop = For::make("Rank", 1, num_processors()-1, ForType::Serial, DeviceAPI::Host, commstmt);

            commloop = Block::make(copy, commloop);

            // Rank 0 sends/recvs; all other ranks issue the complement.
            Stmt body = IfThenElse::make(EQ::make(rank(), 0), commloop, othercommstmt);

            // Allocate scratch
            return allocate_scratch(scratch_name, buffer.type(), b, body);
        }
    }

    // Construct a statement to send the required region of 'buffer'
    // (specified by box 'b') from rank 0 to each processor.
    Stmt send_input_buffer(const AbstractBuffer &buffer, const Box &b) const {
        return communicate_buffer(Send, buffer, b);
    }

    // Construct a statement to receive the region of 'buffer'
    // (specified by box 'b') provided by each processor back to rank
    // 0.
    Stmt recv_output_buffer(const AbstractBuffer &buffer, const Box &b) const {
        return communicate_buffer(Recv, buffer, b);
    }

    // Allocate a buffer of the given name, type, and size that will
    // be used in the given body.
    Stmt allocate_scratch(const string &name, Type type, const Box &b, Stmt body) const {
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

public:
    using IRMutator::visit;

    void visit(const For *for_loop) {
        if (for_loop->for_type != ForType::Distributed) {
            IRMutator::visit(for_loop);
            return;
        }
        // Find all input and output buffers in the loop body
        // using the distributed loop variable.
        FindBuffersUsingVariable find(for_loop->name);
        for_loop->body.accept(&find);

        // Split original loop into chunks of iterations for each rank.
        Stmt newloop = distribute_loop_iterations(for_loop);

        // Get required regions of input buffers in terms of processor
        // rank variable.
        map<string, Box> required, provided;
        required = boxes_required(newloop);
        provided = boxes_provided(newloop);

        // Construct the send statements to send required regions for
        // each input buffer.
        Stmt sendstmt;
        for (const AbstractBuffer &in : find.inputs) {
            Box b = required[in.name()];
            if (sendstmt.defined()) {
                sendstmt = Block::make(sendstmt, send_input_buffer(in, b));
            } else {
                sendstmt = send_input_buffer(in, b);
            }
            ChangeDistributedLoopBuffers change(in.name(), in.partitioned_name(), b);
            newloop = change.mutate(newloop);
        }

        // Construct receive statements to gather output buffer regions
        // back to rank 0.
        Stmt recvstmt;
        for (const AbstractBuffer &out : find.outputs) {
            Box b = provided[out.name()];
            if (recvstmt.defined()) {
                recvstmt = Block::make(recvstmt, recv_output_buffer(out, b));
            } else {
                recvstmt = recv_output_buffer(out, b);
            }
            ChangeDistributedLoopBuffers change(out.name(), out.partitioned_name(), b);
            newloop = change.mutate(newloop);
        }

        newloop = Block::make(sendstmt, Block::make(newloop, recvstmt));

        Stmt allocates;
        for (const AbstractBuffer &in : find.inputs) {
            string scratch_name = in.partitioned_name();
            if (allocates.defined()) {
                allocates = allocate_scratch(scratch_name, in.type(),
                                             required[in.name()], allocates);
            } else {
                allocates = allocate_scratch(scratch_name, in.type(), required[in.name()], newloop);
            }
        }
        for (const AbstractBuffer &out : find.outputs) {
            string scratch_name = out.partitioned_name();
            if (allocates.defined()) {
                allocates = allocate_scratch(scratch_name, out.type(),
                                             provided[out.name()], allocates);
            } else {
                allocates = allocate_scratch(scratch_name, out.type(), provided[out.name()], newloop);
            }
        }

        stmt = LetStmt::make("SliceSize",
                             cast(for_loop->extent.type(), ceil(cast(Float(32), for_loop->extent) / num_processors())),
                             LetStmt::make("Rank", rank(), allocates));
    }

};

Stmt distribute_loops(Stmt s) {
    return DistributeLoops().mutate(s);
}

}
}
