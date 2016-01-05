#ifndef HALIDE_BOUNDS_H
#define HALIDE_BOUNDS_H

/** \file
 * Methods for computing the upper and lower bounds of an expression,
 * and the regions of a function read or written by a statement.
 */

#include "IROperator.h"
#include "Scope.h"

namespace Halide {
namespace Internal {

struct Interval {
    std::string var;
    Expr min, max;
    Interval(std::string var_name, Expr min, Expr max)
        : var(var_name), min(min), max(max) {}
    Interval(std::string var_name) : var(var_name) {}
    Interval() : Interval(unique_name("_interval")) {}
    Interval(Expr min, Expr max) : Interval(unique_name("_interval"), min, max) {}
};

typedef std::map<std::pair<std::string, int>, Interval> FuncValueBounds;

/** Given an expression in some variables, and a map from those
 * variables to their bounds (in the form of (minimum possible value,
 * maximum possible value)), compute two expressions that give the
 * minimum possible value and the maximum possible value of this
 * expression. Max or min may be undefined expressions if the value is
 * not bounded above or below.
 *
 * This is for tasks such as deducing the region of a buffer
 * loaded by a chunk of code.
 */
Interval bounds_of_expr_in_scope(Expr expr,
                                 const Scope<Interval> &scope,
                                 const FuncValueBounds &func_bounds = FuncValueBounds());

/** Represents the bounds of a region of arbitrary dimension. Zero
 * dimensions corresponds to a scalar region. */
struct Box {
    /** The conditions under which this region may be touched. */
    Expr used;

    /** The bounds if it is touched. */
    std::vector<Interval> bounds;

    Box() {}
    Box(size_t sz) : bounds(sz) {}
    Box(const std::vector<Interval> &b) : bounds(b) {}

    size_t size() const {return bounds.size();}
    bool empty() const {return bounds.empty();}
    Interval &operator[](int i) {return bounds[i];}
    const Interval &operator[](int i) const {return bounds[i];}
    void resize(size_t sz) {bounds.resize(sz);}
    void push_back(const Interval &i) {bounds.push_back(i);}

    /** Check if the used condition is defined and not trivially true. */
    bool maybe_unused() const {return used.defined() && !is_one(used);}
    /** Check if the used condition is defined and always false. */
    bool always_unused() const {return used.defined() && is_zero(used);}
};

// Expand box a to encompass box b
void merge_boxes(Box &a, const Box &b);
// Expand box a to encompass box b using halide
void merge_boxes_halide(Box &a, const Box &b);
// Expand box a to encompass box b using nfm
void merge_boxes_nfm(Box &a, const Box &b);

// Test if box a could possibly overlap box b.
bool boxes_overlap(const Box &a, const Box &b);
// Test if box a could possibly overlap box b using halide.
bool boxes_overlap_halide(const Box &a, const Box &b);
// Test if box a could possibly overlap box b using nfm.
bool boxes_overlap_nfm(const Box &a, const Box &b);

// Return expr evaluating whether box a encloses box b.
Expr box_encloses(const Box &a, const Box &b);
// Return expr evaluating whether box a encloses box b using halide.
Expr box_encloses_halide(const Box &a, const Box &b);
// Return expr evaluating whether box a encloses box b using nfm.
Expr box_encloses_nfm(const Box &a, const Box &b);

// Return a Box representing intersection of Box A and Box B
Box boxes_intersection(const Box &a, const Box &b);
// Return a Box representing intersection of Box A and Box B using halide
Box boxes_intersection_halide(const Box &a, const Box &b);
// Return a Box representing intersection of Box A and Box B using nfm
Box boxes_intersection_nfm(const Box &a, const Box &b);

// Return expr evaluating whether box is empty
Expr is_box_empty(const Box &box);
// Return expr evaluating whether box is empty using halide
Expr is_box_empty_halide(const Box &box);
// Return expr evaluating whether box is empty using nfm
Expr is_box_empty_nfm(const Box &box);

/** Compute rectangular domains large enough to cover all the 'Call's
 * to each function that occurs within a given statement or
 * expression. This is useful for figuring out what regions of things
 * to evaluate. */
// @{
std::map<std::string, Box> boxes_required(Expr e,
                                          const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                          const FuncValueBounds &func_bounds = FuncValueBounds());
std::map<std::string, Box> boxes_required(Stmt s,
                                          const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                          const FuncValueBounds &func_bounds = FuncValueBounds());
// @}

/** Compute rectangular domains large enough to cover all the
 * 'Provides's to each function that occurs within a given statement
 * or expression. */
// @{
std::map<std::string, Box> boxes_provided(Expr e,
                                          const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                          const FuncValueBounds &func_bounds = FuncValueBounds());
std::map<std::string, Box> boxes_provided(Stmt s,
                                          const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                          const FuncValueBounds &func_bounds = FuncValueBounds());
// @}

/** Compute rectangular domains large enough to cover all the 'Call's
 * and 'Provides's to each function that occurs within a given
 * statement or expression. */
// @{
std::map<std::string, Box> boxes_touched(Expr e,
                                         const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                         const FuncValueBounds &func_bounds = FuncValueBounds());
std::map<std::string, Box> boxes_touched(Stmt s,
                                         const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                         const FuncValueBounds &func_bounds = FuncValueBounds());
// @}

/** Variants of the above that are only concerned with a single function. */
// @{
Box box_required(Expr e, std::string fn,
                 const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                 const FuncValueBounds &func_bounds = FuncValueBounds());
Box box_required(Stmt s, std::string fn,
                 const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                 const FuncValueBounds &func_bounds = FuncValueBounds());

Box box_provided(Expr e, std::string fn,
                 const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                 const FuncValueBounds &func_bounds = FuncValueBounds());
Box box_provided(Stmt s, std::string fn,
                 const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                 const FuncValueBounds &func_bounds = FuncValueBounds());

Box box_touched(Expr e, std::string fn,
                const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                const FuncValueBounds &func_bounds = FuncValueBounds());
Box box_touched(Stmt s, std::string fn,
                const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                const FuncValueBounds &func_bounds = FuncValueBounds());
// @}

/** Compute the maximum and minimum possible value for each function
 * in an environment. */
FuncValueBounds compute_function_value_bounds(const std::vector<std::string> &order,
                                              const std::map<std::string, Function> &env);

EXPORT void bounds_test();

}
}

#endif
