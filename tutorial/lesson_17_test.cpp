// To run:
//    make tutorial_lesson_17_test

#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Var x("x"), y("y");

    {
        Func f("f"), g("g");
        f(x, y) = x + y;
        g(x, y) = f(x, y);

        Var xi, yi, ti;

        // This results in the f buffer allocated of size 10x10 each iteration of
        // g.tile (of which there are 100). This makes sense to me. However, with
        // this schedule:
        g.tile(x, y, xi, yi, 10, 10).fuse(x, y, ti);
        f.compute_at(g, ti);

        // This results in the f buffer allocated of size 100x10 each iteration of
        // g.tile, of which there are now 50. It seems to me that f should instead
        // be allocated of size 20x10. The above schedule results in the loop nest (I
        // think):
        //g.tile(x, y, xi, yi, 10, 10).fuse(x, y, tile).split(tile, tile, ti, 2);
        //g.tile(x, y, x_inner, y_inner, 10, 10).fuse(x, y, tile_index).split(tile_index, tile_index, ti, 2);
        //f.compute_at(g, tile);

        Image<int> output = g.realize(100, 100);

        printf("Pseudo-code for the schedule:\n");
        g.print_loop_nest();
        printf("\n");
    }

    printf("Success!\n");
    return 0;
}
