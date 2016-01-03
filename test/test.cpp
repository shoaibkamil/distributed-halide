#include "HalideNfmConverter.h"
#include "IR.h"
#include "Bounds.h"
#include "Simplify.h"
#include "IRPrinter.h"
#include "Util.h"
#include "Var.h"
#include "Func.h"

using namespace Halide;
using namespace Halide::Internal;

void bound() {
    Scope<Interval> scope;
    Var x("x"), y("y"), N("N"), K("K"), B("B");
    scope.push("x", Interval(Min::make(Expr(N), Expr(K)), Min::make(Expr(N), Expr(K))));
    scope.push("y", Interval(Min::make(Expr(N-1), Expr(K-1)), Min::make(Expr(N-1), Expr(K-1))));
    Expr z = B*x + B*y;
    FuncValueBounds fb;
    Interval result = bounds_of_expr_in_scope(z, scope, fb);
    std::cout << "Before simplify Result\nmin: " << result.min << "\nmax: " << result.max << "\n";
    if (result.min.defined()) result.min = simplify(result.min);
    if (result.max.defined()) result.max = simplify(result.max);
    std::cout << "After simplify Result\nmin: " << result.min << "\nmax: " << result.max;
}

void tile() {
    Func gradient("gradient_tiled");
    Var B("B"), x("x"), y("y");
    gradient(x, y) = x + y;
    Var x_outer("xo"), x_inner("xi"), y_outer("yo"), y_inner("yi");
    gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, B, B);
    printf("Pseudo-code for the schedule:\n");
    gradient.print_loop_nest();
    printf("\n");
}

void test() {
    Var x("x"), y("y"), c("c"), a("a");
    Func input_16("input_16");
    input_16(x, y, c) = 20;
    //input_16(x, y, c) = cast<uint16_t>(input(x, y, c));

    // Blur it horizontally:
    Func blur_x("blur_x");
    blur_x(x, y, c) = (input_16(x-1, y, c) +
                       2 * input_16(x, y, c) +
                       input_16(x+1, y, c)) / 4;

    // Blur it vertically:
    Func blur_y("blur_y");
    blur_y(x, y, c) = (blur_x(x, y-1, c) +
                       2 * blur_x(x, y, c) +
                       blur_x(x, y+1, c)) / 4;

    // Convert back to 8-bit.
    Func output("output");
    output(x, y, c) = cast<uint8_t>(blur_y(x, y, c));

    Image<uint8_t> result(200, 200, 3);
    result.set_min(1, 1);
    output.realize(result);
}

int main(int argc, const char **argv) {
    ir_nfm_test();
    ////bound();
    //tile();
    //test();
    return 0;
}
