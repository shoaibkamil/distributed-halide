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
const bool profiling = false;
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

vector<Expr> dims_to_print(const Box &b, const string &name) {
    vector<Expr> result;
    for (unsigned i = 0; i < b.size(); i++) {
        result.push_back(string("\n   ") + name + "[" + std::to_string(i) + "].min =");
        result.push_back(b[i].min);
        result.push_back(name + "[" + std::to_string(i) + "].max =");
        result.push_back(b[i].max);
    }
    return result;
}

// Return expr evaluating whether box a encloses box b.
Expr box_encloses(const Box &a, const Box &b) {
    internal_assert(a.size() == b.size());
    Expr e = simplify(a[0].min <= b[0].min && a[0].max >= b[0].max);
    for (unsigned i = 1; i < a.size(); i++) {
        e = e && simplify(a[i].min <= b[i].min && a[i].max >= b[i].max);
    }
    return simplify(e);
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
Box simplify_box(const Box &b, const vector<std::pair<string, Expr> > &lets) {
    Box result;
    // TODO: this can be quite slow with large
    // environments. The goal is to get the Box into terms of
    // the outermost variables (the global output buffer
    // variables).
    for (unsigned i = 0; i < b.size(); i++) {
        Expr min = b[i].min, max = b[i].max;
        for (auto let = lets.rbegin(); let != lets.rend(); ++let) {
            min = simplify(substitute(let->first, let->second, min));
            max = simplify(substitute(let->first, let->second, max));
        }
        result.push_back(Interval(min, max));
    }
    return result;
}

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
        _distributed = buffer.distributed();
        _dimensions = buffer.dimensions();
        for (int i = 0; i < buffer.dimensions(); i++) {
            if (_distributed) {
                Expr min = buffer.allocated_min(i);
                Expr max = min + buffer.allocated_extent(i) - 1;
                _shape.push_back(Interval(min, max));
                // These are parameterized by rank and number of processors.
                Expr havemin = buffer.local_min(i);
                Expr havemax = havemin + buffer.local_extent(i) - 1;
                _bounds.push_back(Interval(havemin, havemax));
            } else {
                Expr min = buffer.min(i);
                Expr max = min + buffer.extent(i) - 1;
                _shape.push_back(Interval(min, max));
                _bounds.push_back(Interval(min, max));
            }
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

    // Return the size of the given box in bytes according to the type
    // of this buffer.
    Expr size_of(const Box &b) const {
        internal_assert(b.size() > 0);
        Expr num_elems = make_one(Int(64));
        for (unsigned i = 0; i < b.size(); i++) {
            num_elems *= cast(Int(64), simplify(b[i].max - b[i].min) + 1);
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
        internal_assert(!is_input_image());
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
        Box local_origin = shape();
        for (unsigned i = 0; i < b.size(); i++) {
            result[i] = Interval(simplify(b[i].min - local_origin[i].min),
                                 simplify(b[i].max - local_origin[i].min));
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
    return communicate_subarray(Recv, buf.name(), buf.type(), shape, b, rank);
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

// Used for profiling
const string profile_buf = "DistrProfileBuffer";
const string profile_gathered_buf = "DistrProfileBufferGathered";
map<string, int> profile_indices;   // map name -> index in buffer.

// Return current time in nanoseconds.
Expr profile_time() {
    // Pass a dummy unique "id" value as argument. This is to prevent
    // multiple calls from being folded into one (Call node equality
    // just looks at type, name, and args).
    static int id = 0;
    return Call::make(UInt(64), "halide_distr_time_ns", std::vector<Expr>({id++}), Call::Extern);
}

// Get or assign an index into the profiling buffer for the given
// event name.
Expr get_profiling_index(const string& s) {
    if (profile_indices.find(s) == profile_indices.end()) {
        int idx = profile_indices.size();
        profile_indices[s] = idx;
    }
    return profile_indices[s];
}

// Add profiling (timing) arround the given statement. The event name
// is constructed as operation.name
Stmt add_profiling(Stmt s, const string &operation, const string &name) {
    const string id = operation + "." + name;
    const string start_name = "start_timer_" + id;
    Expr idx = get_profiling_index(id);
    Expr start = Variable::make(UInt(64), start_name);
    Expr old_val = Load::make(UInt(64), profile_buf, idx, Buffer(), Parameter());
    Expr delta = Sub::make(profile_time(), start);
    Expr new_val = Add::make(old_val, delta);
    s = Block::make(s, Store::make(profile_buf, new_val, idx));
    s = LetStmt::make(start_name, profile_time(), s);
    return s;
}

// Gather all profiling data to rank 0 into a larger "gathered" buffer.
Stmt gather_profiling() {
    Stmt rank0copy;
    // Copy rank 0 profiling into gathered buffer
    Expr stride = (int)profile_indices.size();
    for (const std::pair<std::string, int> &i : profile_indices) {
        int eventidx = i.second;
        Expr val = Load::make(UInt(64), profile_buf, eventidx, Buffer(), Parameter());
        Expr idx = eventidx + rank() * stride;
        Stmt store = Store::make(profile_gathered_buf, val, idx);
        if (rank0copy.defined()) {
            rank0copy = Block::make(rank0copy, store);
        } else {
            rank0copy = store;
        }
    }
    // Receive other ranks into gathered buffer
    Expr addr = address_of(profile_gathered_buf, Var("r") * stride * UInt(64).bytes());
    Expr count = (int)profile_indices.size();
    Stmt recvstmt = Evaluate::make(recv(addr, UInt(64), count, Var("r")));
    Stmt recvloop = For::make("r", 1, Var("NumProcessors")-1, ForType::Serial, DeviceAPI::Host, recvstmt);
    recvloop = Block::make(rank0copy, recvloop);

    // Send profiling to rank 0
    addr = address_of(profile_buf, 0);
    Stmt sendstmt = Evaluate::make(send(addr, UInt(64), count, 0));

    return IfThenElse::make(rank() == 0, recvloop, sendstmt);
}

// Insert statements to print all profiling data. Rank 0 gathers all
// data from other ranks and prints: other ranks do not print.
Stmt print_profiling(Stmt s) {
    Stmt gather = gather_profiling();
    Expr stride = (int)profile_indices.size();
    Stmt printstmt;
    for (const std::pair<std::string, int> &i : profile_indices) {
        int eventidx = i.second;
        Expr idx = eventidx + Var("r") * stride;
        Expr val = Load::make(UInt(64), profile_gathered_buf, idx, Buffer(), Parameter());
        vector<Expr> msg = {string("rank"), Var("r"), string("profiling"),
                            i.first, string("="), val, string("ns")};
        if (printstmt.defined()) {
            printstmt = Block::make(printstmt, Evaluate::make(print(msg)));
        } else {
            printstmt = Evaluate::make(print(msg));
        }
    }
    Stmt printloop = IfThenElse::make(rank() == 0, For::make("r", 0, Var("NumProcessors"), ForType::Serial, DeviceAPI::Host, printstmt));

    s = Block::make(s, gather);
    s = Block::make(s, printloop);

    s = Allocate::make(profile_gathered_buf, UInt(64), {stride, Var("NumProcessors")}, const_true(), s);
    return s;
}

// Allocate a profiling buffer for the events on the current rank.
Stmt allocate_profiling(Stmt s) {
    Expr i = Variable::make(Int(32), "i");
    Stmt init = For::make("i", 0, (int)profile_indices.size(), ForType::Serial, DeviceAPI::Host,
                          Store::make(profile_buf, Cast::make(UInt(64), 0), i));
    s = Block::make(init, s);
    s = Allocate::make(profile_buf, UInt(64), {(int)profile_indices.size()}, const_true(), s);
    return s;
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
    Expr ghost_zone_empty;

    switch (cmd) {
    case Send:
        I = BoxIntersection(have, need_parameterized);
        ghost_zone_empty = box_encloses(have_parameterized, need_parameterized);
        break;
    case Recv:
        I = BoxIntersection(have_parameterized, need);
        ghost_zone_empty = box_encloses(have, need);
        break;
    }

    //ghost_zone_empty = const_false();

    Expr addr;
    Expr numbytes = Variable::make(Int(64), "msgsize");
    Expr cond = And::make(NE::make(Var("Rank"), Var("r")), And::make(numbytes > 0, Not::make(ghost_zone_empty)));
    Stmt commstmt;

    // Convert the intersection box to "local" coordinates (the
    // allocated buffer counts from 0). This just means subtracting the
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
            addr = address_of(buf.name(), local_need[0].min * buf.elem_size());
            Expr msgsize = local_have[0].max - local_have[0].min + 1;
            commstmt = IfThenElse::make(cond, Evaluate::make(irecv(addr, buf.type(), msgsize, Var("r"))));
        } else {
            commstmt = irecv_subarray(buf, buf.shape(), local_need, Var("r"));
        }
        break;
    }

    Box shape = buf.shape();
    if (trace_messages && cmd == Send) {
        vector<Expr> msg = {string("rank"), rank(), string("sending to rank"), Var("r"),
                            string("buffer " + buf.name() + ":\n"), string("size")};
        vector<Expr> boxstr = dims_to_print(I.box(), "I.box()");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(shape, "shape");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(have, "have");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(need, "need");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(need_parameterized, "need_parameterized");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(local_have, "local_have");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());

        Stmt p = Evaluate::make(print_when(cond, msg));
        commstmt = Block::make(p, commstmt);
    }

    if (trace_messages && cmd == Recv) {
        vector<Expr> msg = {string("rank"), rank(), string("receiving from rank"), Var("r"),
                            string("buffer " + buf.name() + ":\n"), string("size")};
        vector<Expr> boxstr = dims_to_print(I.box(), "I.box()");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(shape, "shape");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(have, "have");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(need, "need");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(have_parameterized, "have_parameterized");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());
        boxstr = dims_to_print(local_need, "local_need");
        msg.insert(msg.end(), boxstr.begin(), boxstr.end());

        Stmt p = Evaluate::make(print_when(cond, msg));
        commstmt = Block::make(p, commstmt);
    }

    commstmt = IfThenElse::make(cond, commstmt);
    commstmt = LetStmt::make("msgsize", buf.size_of(I.box()), commstmt);

    // Compute the "maximal need" region, i.e. the largest need region
    // of all ranks. We do this by simply calculating the need region
    // of the middle rank, which by construction will be maximal
    // (ranks closer to the edge may need a smaller region due to
    // boundary conditions). TODO: This approach currently doesn't
    // extend to boundary conditions beyond simple
    // clamp-to-edge. Extending to the other extreme (tiling the image
    // at the boundary) is a fairly straightforward extension. See
    // notes from 10/6/15.
    // Expr middle_rank = cast(Int(32), ceil(cast(Float(32), Var("NumProcessors")) / 2.0f));
    // env.ref("Rank") = middle_rank;
    // Box maximal_need = simplify_box(need, env);

    // // Compute the "rank span" using the have/need regions. This is
    // // the rank "radius" that we must communicate with.
    // Expr k = 0;
    // for (unsigned i = 0; i < maximal_need.size(); i++) {
    //     Expr hext = have[i].max - have[i].min + 1,
    //         next = maximal_need[i].max - maximal_need[i].min + 1;
    //     Expr kk = cast(Int(32), ceil(cast(Float(32), next) / hext));
    //     k = max(k, kk);
    // }

    // Expr left, right;
    // left = max(Var("Rank") - k, 0);
    // right = min(Var("Rank") + k, Var("NumProcessors") - 1);
    // Expr rankextent = right - left + 1;
    // commstmt = For::make("r", left, rankextent, ForType::Serial, DeviceAPI::Host, commstmt);
    commstmt = For::make("r", 0, Var("NumProcessors"), ForType::Serial, DeviceAPI::Host, commstmt);
    return commstmt;
}

// For each required region, generate communication code between ranks
// that own data needed by other ranks.
Stmt exchange_data(const string &func, const vector<AbstractBuffer> &required, Stmt &sendwait_out) {
    Stmt sendloop, recvloop;
    Stmt sendwait, recvwait;
    for (const auto it : required) {
        const AbstractBuffer &in = it;
        // No border exchanges needed for non-distributed buffers or
        // output images.
        if (!in.distributed() || in.is_output_image()) continue;

        Stmt sendstmt = communicate_intersection(Send, in, func);
        Stmt sendwaitstmt = waitall_isend(in.name());
        Stmt recvstmt = communicate_intersection(Recv, in, func);
        Stmt recvwaitstmt = waitall_irecv(in.name());
        if (profiling) {
            sendstmt = add_profiling(sendstmt, "border_exchange_send", func + "." + in.name());
            sendwaitstmt = add_profiling(sendwaitstmt, "border_exchange_send", func + "." + in.name());
            recvstmt = add_profiling(recvstmt, "border_exchange_recv", func + "." + in.name());
            recvwaitstmt = add_profiling(recvwaitstmt, "border_exchange_recv", func + "." + in.name());
        }

        if (sendloop.defined()) {
            sendloop = Block::make(sendloop, sendstmt);
            sendwait = Block::make(sendwait, sendwaitstmt);
        } else {
            sendloop = sendstmt;
            sendwait = sendwaitstmt;
        }

        if (recvloop.defined()) {
            recvloop = Block::make(recvloop, recvstmt);
            recvwait = Block::make(recvwait, recvwaitstmt);
        } else {
            recvloop = recvstmt;
            recvwait = recvwaitstmt;
        }
    }
    internal_assert(sendloop.defined() == recvloop.defined());
    if (!sendloop.defined()) {
        return Stmt();
    } else {
        Stmt comm = Block::make(recvloop, sendloop);
        comm = Block::make(comm, recvwait);
        sendwait_out = sendwait;
        return comm;
    }
}

// Change all uses of the original input buffers to use "local"
// indices (instead of global).
Stmt update_io_buffers(Stmt loop, const string &func, const vector<AbstractBuffer> &required,
                       const vector<AbstractBuffer> &provided) {
    for (const auto it : required) {
        const AbstractBuffer &in = it;
        const Box &b = in.shape();
        if (!in.is_image()) continue;
        if (in.is_input_image()) {
            ChangeDistributedLoopBuffers change(in.name(), in.name(), b, true);
            loop = change.mutate(loop);
        } else {
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

        Stmt sendwait;
        Stmt border_exchange = exchange_data(name, required, sendwait);
        if (border_exchange.defined()) {
            // No overlap of communication and computation:
            border_exchange = Block::make(border_exchange, sendwait);
            newstmt = Block::make(border_exchange, newstmt);
            // Overlap sends with computation (slower on local_laplacian):
            //newstmt = Block::make(newstmt, sendwait);
        }

        if (trace_progress) {
            Stmt p = Evaluate::make(print({string("rank"), rank(), string("stage"), name,
                            string("before exchange_data")}));
            newstmt = Block::make(p, newstmt);
            p = Evaluate::make(print({string("rank"), rank(), string("stage"), name,
                            string("after exchange_data")}));
            newstmt = Block::make(newstmt, p);
        }

        newstmt = update_io_buffers(newstmt, name, required, provided);
        return newstmt;
    }

public:
    const map<string, AbstractBuffer> &buffers;
    InjectCommunication(const map<string, AbstractBuffer> &bufs) : buffers(bufs) {}

    using IRMutator::visit;

    void visit(const ProducerConsumer *op) {
        string current_function = op->name;
        Stmt newproduce = op->produce, newupdate = op->update;
        if (profiling) {
            newproduce = add_profiling(newproduce, "compute", op->name);
            if (newupdate.defined()) newupdate = add_profiling(newupdate, "compute", op->name);
        }
        newproduce = inject_communication(op->name, newproduce);
        stmt = ProducerConsumer::make(op->name, newproduce,
                                      newupdate.defined() ? inject_communication(op->name, newupdate) : mutate(newupdate),
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

// Sets distributed loop bounds for any functions marked
// compute_rank(). Must occur after DistributeLoops.
class LowerComputeRankFunctions : public IRMutator {
    const map<string, Function> &env;
    map<string, Box> required_regions;

    class MergeAllRequiredRegions : public IRVisitor {
        const FuncValueBounds &func_bounds;
        Scope<Interval> scope;
        vector<std::pair<string, Expr> > lets;
    public:
        map<string, Box> regions;

        MergeAllRequiredRegions(const FuncValueBounds &fb) : func_bounds(fb) {}

        using IRVisitor::visit;

        void visit(const For *for_loop) {
            scope.push(for_loop->name, Interval(for_loop->min, for_loop->min + for_loop->extent - 1));
            IRVisitor::visit(for_loop);
            scope.pop(for_loop->name);
        }

        void visit(const LetStmt *let) {
            Interval I = bounds_of_expr_in_scope(let->value, scope, func_bounds);
            internal_assert(I.min.defined() == I.max.defined());
            if (I.min.defined()) {
                scope.push(let->name, I);
            }
            lets.push_back(std::make_pair(let->name, let->value));
            IRVisitor::visit(let);
            if (I.min.defined()) {
                scope.pop(let->name);
            }
            lets.pop_back();
        }

        void visit(const Let *let) {
            Interval I = bounds_of_expr_in_scope(let->value, scope, func_bounds);
            internal_assert(I.min.defined() == I.max.defined());
            if (I.min.defined()) {
                scope.push(let->name, I);
            }
            lets.push_back(std::make_pair(let->name, let->value));
            IRVisitor::visit(let);
            if (I.min.defined()) {
                scope.pop(let->name);
            }
            lets.pop_back();
        }

        void visit(const Call *op) {
            map<string, Box> required = boxes_required(op, scope, func_bounds);
            for (auto it : required) {
                string name = it.first;
                if (regions.find(name) != regions.end()) {
                    Box b = regions[name];
                    merge_boxes(b, it.second);
                    regions[name] = simplify_box(b, lets);
                } else {
                    regions[name] = simplify_box(it.second, lets);
                }
            }
        }
    };

    // For a loop variable corresponding to a function argument (not a
    // split loop variable), return an Interval containing the min and
    // max values for the loop variable based on the required region
    // of the function by its consumers.
    Interval get_required_interval(const string &func, const string &loop_var) const {
        internal_assert(required_regions.find(func) != required_regions.end());
        internal_assert(env.find(func) != env.end());
        const vector<string> &args = env.at(func).args();
        const Box &b = required_regions.at(func);
        internal_assert(b.size() == args.size());
        for (int i = 0; i < (int)args.size(); i++) {
            if (ends_with(loop_var, args[i])) {
                return b[i];
            }
        }
        internal_assert(false) << loop_var;
        return Interval(0, 0);
    }

    // Returns true if the given name is a split (or fuse) dimension
    // of the given function.
    bool var_is_split(const string &func, const string &var) const {
        for (const Split &sp : env.at(func).schedule().splits()) {
            if (ends_with(var, sp.outer) || ends_with(var, sp.inner)) {
                return true;
            }
        }
        return false;
    }

    bool var_is_loop_bound(const string &var) const {
        return ends_with(var, ".loop_min") ||
            ends_with(var, ".loop_max") ||
            ends_with(var, ".loop_extent");
    }

public:
    LowerComputeRankFunctions(Stmt s, const std::map<std::string, Function> &e, const FuncValueBounds &func_bounds) : env(e) {
        MergeAllRequiredRegions find(func_bounds);
        s.accept(&find);
        required_regions.swap(find.regions);
    }

    using IRMutator::visit;

    void visit(const LetStmt *let) {
        string loop_var = remove_suffix(let->name);
        string funcname = first_token(let->name);
        string stage_prefix = funcname + "." + second_token(let->name);
        auto it = env.find(funcname);
        if (it != env.end() && it->second.schedule().compute_level().is_rank() &&
            !var_is_split(funcname, loop_var) && var_is_loop_bound(let->name)) {
            Interval I = get_required_interval(funcname, loop_var);
            if (ends_with(let->name, ".loop_min")) {
                stmt = LetStmt::make(let->name, I.min, mutate(let->body));
            } else if (ends_with(let->name, ".loop_max")) {
                stmt = LetStmt::make(let->name, I.max, mutate(let->body));
            } else if (ends_with(let->name, ".loop_extent")) {
                stmt = LetStmt::make(let->name, I.max - I.min + 1, mutate(let->body));
            } else {
                internal_assert(false) << let->name;
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

    ForType get_new_type(ForType t) const {
        if (t == ForType::Distributed) {
            return ForType::Serial;
        } else if (t == ForType::DistributedParallel) {
            return ForType::Parallel;
        } else {
            return t;
        }
    }

    void visit(const For *for_loop) {
        IRMutator::visit(for_loop);
        ForType newtype = get_new_type(for_loop->for_type);
        if (newtype != for_loop->for_type) {
            stmt = For::make(for_loop->name, for_loop->min, for_loop->extent,
                             newtype, for_loop->device_api, for_loop->body);
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
            // An output won't have a Realize node, so we have to set
            // the shape here.
            Box b = box_touched(op->produce, op->name);
            buf.set_shape(b);
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

Stmt distribute_loops(Stmt s, const std::map<std::string, Function> &env, const FuncValueBounds &func_bounds, bool cap_extents) {
    FindDistributedLoops find;
    s.accept(&find);
    if (find.distributed_functions.empty()) {
        return s;
    }
    s = DistributeLoops(find.distributed_bounds, env, cap_extents).mutate(s);
    s = LowerComputeRankFunctions(s, env, func_bounds).mutate(s);
    return s;
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
    if (profiling) {
        s = print_profiling(s);
        s = allocate_profiling(s);
    }
    s = ChangeDistributedFor().mutate(s);
    s = LetStmt::make("Rank", rank(), s);
    s = LetStmt::make("NumProcessors", num_processors(), s);
    return s;
}

Stmt change_distributed_annotation(Stmt s) {
    return ChangeDistributedFor().mutate(s);
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

Box offset_box(const Box &b, const vector<Expr> &offset) {
    internal_assert(b.size() == offset.size());
    Box result(b.size());
    for (unsigned i = 0; i < b.size(); i++) {
        result[i] = Interval(b[i].min - offset[i], b[i].max - offset[i]);
    }
    return result;
}

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
    bool any_memoized = false;
    Stmt s = schedule_functions(outputs, order, env, any_memoized, !t.has_feature(Target::NoAsserts));
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

int64_t expr2int(Expr e) {
    const int64_t *result = as_const_int(simplify(e));
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
