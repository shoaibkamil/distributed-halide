#include "HalideNfmConverter.h"
#include "NfmToHalide.h"

#include "Bounds.h"
#include "IR.h"
#include "IREquality.h"
#include "Simplify.h"
#include "Substitute.h"
#include "IRPrinter.h"
#include "Util.h"
#include "Var.h"
#include "Func.h"

using namespace Halide;
using namespace Halide::Internal;
using namespace Nfm;
using namespace Nfm::Internal;

using std::string;
using std::vector;

Expr simplify_expr(Expr expr, vector<string>& loop_dims) {
    std::cout << "Start simplifying expression: " << expr << "\n";

    CollectVars collect(loop_dims);
    collect.mutate(expr);

    std::cout << "Dimensions: ";
    const auto& dims = collect.get_dims();
    for (size_t i = 0; i < dims.size(); ++i) {
        std::cout << dims[i];
        if (i != dims.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";

    std::cout << "Symbolic Constants: ";
    const auto& sym_consts = collect.get_sym_consts();
    for (size_t i = 0; i < sym_consts.size(); ++i) {
        std::cout << sym_consts[i];
        if (i != sym_consts.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";

    std::cout << "Let Assignments: ";
    const auto& let_assignments = collect.get_let_assignments();
    for (size_t i = 0; i < let_assignments.size(); ++i) {
        std::cout << let_assignments[i];
        if (i != let_assignments.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";

    std::map<std::string, Expr> expr_substitutions;
    std::vector<std::pair<std::string, Expr>> let_substitutions;
    NfmUnionDomain union_dom = convert_halide_expr_to_nfm_union_domain(
        expr, collect.get_sym_consts(), collect.get_dims(), &expr_substitutions, &let_substitutions);
    //std::cout << "\nNfmUnionDomain: " << union_dom << "\n\n";

    std::cout << "\nSubstitutions: ";
    for (auto& iter : expr_substitutions) {
        std::cout << "FROM " << iter.first << " TO: " << iter.second << "\n";
    }
    std::cout << "\n";

    Expr simplified_expr = convert_nfm_union_domain_to_halide_expr(
        Int(32), union_dom, &let_assignments, &expr_substitutions, &let_substitutions);
    std::cout << "\n\nSimplified expr: " << simplified_expr << "\n\n";

    Interval result =
        convert_nfm_union_domain_to_halide_interval(Int(32), union_dom, "w",
            &let_assignments, &expr_substitutions, &let_substitutions);
    std::cout << "\n\nSimplified interval:\n"
              << "  Min: " << result.min << "\n"
              << "  Max: " << result.max << "\n\n";

    /*NfmUnionDomain union_dom2 = convert_halide_expr_to_nfm_union_domain(
        simplified_expr, collect.get_sym_consts(), collect.get_dims());
    //std::cout << "\nNfmUnionDomain: " << union_dom2 << "\n\n";

    Expr simplified_expr2 = convert_nfm_union_domain_to_halide_expr(Int(32), union_dom2);
    std::cout << "\n\nSimplified expr: " << simplified_expr2 << "\n\n";

    Interval result2 =
        convert_nfm_union_domain_to_halide_interval(Int(32), union_dom2, "w", &let_assignments);
    std::cout << "\n\nSimplified interval:\n"
              << "  Min: " << result2.min << "\n"
              << "  Max: " << result2.max << "\n\n";

    user_assert(equal(result.min, result2.min)) << "Both intervals should have the same lower bound\n"
        << "  result.min: " << result.min << "\n  result2.min: " << result2.min << "\n";
    user_assert(equal(result.max, result2.max)) << "Both intervals should have the same upper bound\n"
        << "  result.max: " << result.max << "\n  result2.max: " << result2.max << "\n";*/
    return Expr();
}

Interval simplify_interval(Interval interval, vector<string>& sym_consts,
                           vector<string>& dims) {
    std::cout << "Start simplifying interval:\n"
              << "  Min: " << interval.min << "\n"
              << "  Max: " << interval.max << "\n";

    Expr expr = convert_interval_to_expr(interval);
    CollectVars collect(dims);
    collect.mutate(expr);

    std::cout << "Dimensions: ";
    const auto& c_dims = collect.get_dims();
    //assert(dims == c_dims);
    for (size_t i = 0; i < c_dims.size(); ++i) {
        std::cout << c_dims[i];
        if (i != c_dims.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";

    std::cout << "Symbolic Constants: ";
    const auto& c_sym_consts = collect.get_sym_consts();
    //assert(sym_consts == c_sym_consts);
    for (size_t i = 0; i < c_sym_consts.size(); ++i) {
        std::cout << c_sym_consts[i];
        if (i != c_sym_consts.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";

    std::cout << "Let Assignments: ";
    const auto& let_assignments = collect.get_let_assignments();
    for (size_t i = 0; i < let_assignments.size(); ++i) {
        std::cout << let_assignments[i];
        if (i != let_assignments.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";

    std::map<std::string, Expr> expr_substitutions;
    std::vector<std::pair<std::string, Expr>> let_substitutions;
    NfmUnionDomain union_dom = convert_halide_interval_to_nfm_union_domain(
        interval, sym_consts, dims, &expr_substitutions, &let_substitutions);
    //std::cout << "\nNfmUnionDomain: " << union_dom << "\n\n";

    std::cout << "\nSubstitutions: \n";
    for (auto& iter : expr_substitutions) {
        std::cout << "FROM " << iter.first << " TO: " << iter.second << "\n";
    }
    std::cout << "\n";

    Interval result = convert_nfm_union_domain_to_halide_interval(
        Int(32), union_dom, interval.var, &let_assignments, &expr_substitutions, &let_substitutions);
    std::cout << "Simplified interval:\n"
              << "  Min: " << result.min << "\n"
              << "  Max: " << result.max << "\n\n";

    std::map<std::string, Expr> expr_substitutions2;
    std::vector<std::pair<std::string, Expr>> let_substitutions2;
    NfmUnionDomain union_dom2 = convert_halide_interval_to_nfm_union_domain(
        result, sym_consts, dims, &expr_substitutions2, &let_substitutions2);
    //std::cout << "\nNfmUnionDomain: " << union_dom << "\n\n";

    std::cout << "\nSubstitutions 2: ";
    for (auto& iter : expr_substitutions2) {
        std::cout << "FROM " << iter.first << " TO: " << iter.second << "\n";
    }
    std::cout << "\n";

    Interval result2 = convert_nfm_union_domain_to_halide_interval(
        Int(32), union_dom2, result.var, &let_assignments, &expr_substitutions2, &let_substitutions2);
    std::cout << "Simplified interval:\n"
              << "  Min: " << result2.min << "\n"
              << "  Max: " << result2.max << "\n\n";
    user_assert(equal(result.min, result2.min)) << "Both intervals should have the same lower bound\n"
        << "  result.min: " << result.min << "\n  result2.min: " << result2.min << "\n";
    user_assert(equal(result.max, result2.max)) << "Both intervals should have the same upper bound\n"
        << "  result.max: " << result.max << "\n  result2.max: " << result2.max << "\n";
    return result;
}

void example_expr() {
    Expr s = Variable::make(Int(32), "s");
    Expr x = Variable::make(Int(32), "x");
    Expr y = Variable::make(Int(32), "y");
    Expr z = Variable::make(Int(32), "z");
    Expr w = Variable::make(Int(32), "w");
    Expr M = Variable::make(Int(32), "M");
    Expr N = Variable::make(Int(32), "N");
    Expr P = Variable::make(Int(32), "P");

    vector<string> loop_dims = {"s", "x", "y", "z", "w"};
    //Expr expr = EQ::make(2*w, min(min(4*x, 3*y), 6*y+7*z));
    //Expr expr = EQ::make(2*w, x);
    //Expr expr = Let::make("N", cast(Int(32), ceil(1000.000000f/cast(Float(32), M))), EQ::make(w, N*x+y));
    //Expr expr = Let::make("y", cast(Int(32), ceil(1000.000000f/cast(Float(32), M))), EQ::make(w, x+y));
    /*Expr expr = Let::make("y", cast(Int(32), ceil(1000.000000f/cast(Float(32), M))),
                    Let::make("x", cast(Int(32), ceil(20.0f/cast(Float(32), N))), EQ::make(w, x+y)));*/
    //Expr expr = Let::make("N", min(10, P), Let::make("M", min(N, 3*N-P), EQ::make(w, M*x+N*y)));
    //Expr expr = Let::make("N", min(10, P), EQ::make(w, M*x+N*y));
    //Expr expr = GE::make(w, select(10 < P, y*10+M*x, P*y+M*x));
    //Expr expr = LE::make(w, Let::make("N", min(10, P), M*x+N*y));
    //Expr expr = EQ::make(w, max(min(x, y), z) + 1 - min(min(z, y-1), 3));
    //Expr expr = w >= max(x,y)-1;
    //Expr expr = w >= max(max(x,y)+1, z);
    //Expr expr = w >= select((((-1 - z) >= 0) && (x >= 0)), (x*2), select((((-1 - z) >= 0) && ((-1 - x) >= 0)), (y + x), select(((z >= 0) && (x >= 0)), (z + x), (z + y))));
    //Expr expr = w >= select(x>=0 && y>=0, M, N);
    //Expr expr = w >= select((((-1 - z) >= 0) && ((-1 - x) >= 0)), (y + x), select(((z >= 0) && (x >= 0)), (z + x), (z + y)));
    Expr expr = w >= select(((z >= 0) && (x >= 0)), (z + x), (z + y));
    //Expr expr = (z>=0 || x<0) && (z<0 && x<0);
    Expr simplified_expr = simplify_expr(expr, loop_dims);
}

void example_interval() {
    Expr x = Variable::make(Int(32), "x");
    Expr y = Variable::make(Int(32), "y");
    Expr z = Variable::make(Int(32), "z");
    Expr w = Variable::make(Int(32), "w");
    Expr M = Variable::make(Int(32), "M");
    Expr N = Variable::make(Int(32), "N");
    Expr P = Variable::make(Int(32), "P");

    vector<string> sym_consts = {"P", "N", "M"};
    vector<string> dims = {"x", "y", "z", "w"};
    //Interval interval("w", min((2*x+3)/2, z), max(2*y/2, 4*z/2));
    //Interval interval("w", min(x+y, x+z), max(y, z));
    //Interval interval("w", select(x>=0, x, select(y>=0, y, z)), Expr());
    //Interval interval("w", x, cast(Int(32), 10.0f*x));
    //Interval interval("w", x, min(y, z));
    //Interval interval("w", cast(Int(32), 10.0f*x), cast(Int(32), 10.0f*x));
    //Interval interval("w", Let::make("N", min(10, P), M*x+N*y), Let::make("N", min(10, P), M*x+N*y));
    //Interval interval("w", z, Let::make("y", min(10, P+M), 3*y));

    // Note: Halide Simplify does not simplify ((P + P)*3) into P*6
    //Interval interval("w", Let::make("y", min(10, P+M), 3*y), Let::make("N", min(10, P), Let::make("M", min(N, 3*N-P), M*x+N*y)));

    // Note: the resulting interval result1 and result2's upperbounds are not exactly of
    // the same form, but they are equal (ignore the assertion error)
    //Interval interval("w", x, Let::make("N", min(10, P), Let::make("M", min(N, 3*N-P), M*x+N*y)));

    //Interval interval("w", select(x>=0, x, y) + select(z>=0, z, x), Expr());
    //Interval interval("w", min(select((((M + 1) <= N) && (M <= 3)), M, select((min(M, 4) < N), 3, (N + -1))), P), Expr());
    //Interval interval("w", select(min(M, N) >= 0, x, y), Expr());
    Interval interval("w", max(x, y), Expr());

    //Interval interval("w", Expr(), min((max(M, (N + -1)) + 0), 3));
    //Interval interval("w", Expr(), select(x < y, min(y, 3), x));
    Interval simplified_interval = simplify_interval(interval, sym_consts, dims);
}

void boxes_overlap_test() {
    Expr x = Variable::make(Int(32), "x");
    Expr y = Variable::make(Int(32), "y");
    Expr z = Variable::make(Int(32), "z");
    Expr w = Variable::make(Int(32), "w");
    Expr M = Variable::make(Int(32), "M");
    Expr N = Variable::make(Int(32), "N");
    Expr P = Variable::make(Int(32), "P");

    vector<string> sym_consts = {"P", "N", "M"};
    vector<string> dims = {"x", "y", "z", "w"};

    Box a, b;
    a.push_back(Interval("x", 1, min(M, 3)));
    a.push_back(Interval("y", N, 6));
    a.push_back(Interval("z", 0, 4));

    b.push_back(Interval("x", 0, N));
    b.push_back(Interval("y", 0, 4));
    b.push_back(Interval("z", 0, 4));

    bool is_overlap = boxes_overlap_nfm(a, b);
    printf("is overlap? %d\n", is_overlap);

    Box intersection = boxes_intersection_nfm(a, b);
    std::cout << "Box intersection NFM:\n";
    for (size_t i = 0; i < intersection.size(); ++i) {
        std::cout << "Dim: " << intersection[i].var << "\n  min: " << a[i].min
                  << "\n  max: " << intersection[i].max << "\n";
    }
    Expr empty_intersection = is_box_empty_nfm(intersection);
    std::cout << "Box intersection NFM is empty? " << empty_intersection << "\n";

    Box intersection_halide = boxes_intersection_halide(a, b);
    std::cout << "\nBox intersection Halide:\n";
    for (size_t i = 0; i < intersection_halide.size(); ++i) {
        std::cout << "Dim: " << intersection_halide[i].var << "\n  min: " << a[i].min
                  << "\n  max: " << intersection_halide[i].max << "\n";
    }
    Expr empty_intersection_halide = is_box_empty_halide(intersection);
    std::cout << "Box intersection Halide is empty? " << empty_intersection_halide << "\n";
}

void box_encloses_test() {
    Expr x = Variable::make(Int(32), "x");
    Expr y = Variable::make(Int(32), "y");
    Expr z = Variable::make(Int(32), "z");
    Expr w = Variable::make(Int(32), "w");
    Expr M = Variable::make(Int(32), "M");
    Expr N = Variable::make(Int(32), "N");
    Expr P = Variable::make(Int(32), "P");

    vector<string> sym_consts = {"P", "N", "M"};
    vector<string> dims = {"x", "y", "z", "w"};

    Box a, b;
    a.push_back(Interval("x", 1, M*x));
    a.push_back(Interval("y", 1, N));
    a.push_back(Interval("z", 0, 4));

    b.push_back(Interval("x", x, 4));
    b.push_back(Interval("y", N, 3));
    b.push_back(Interval("z", 11, 14));

    Expr result = box_encloses_nfm(a, b);
    std::cout << "A encloses B? " << result << "\n";
}

void boxes_merge_test() {
    Expr x = Variable::make(Int(32), "x");
    Expr y = Variable::make(Int(32), "y");

    Expr M = Variable::make(Int(32), "M");
    Expr N = Variable::make(Int(32), "N");
    Expr P = Variable::make(Int(32), "P");
    Expr Q = Variable::make(Int(32), "Q");
    Expr R = Variable::make(Int(32), "R");
    Expr S = Variable::make(Int(32), "S");

    vector<string> sym_consts = {"P", "N", "M"};
    vector<string> dims = {"x", "y"};

    Box a, b;
    //a.push_back(Interval("x"));
    //a.push_back(Interval("y", P, Q));

    b.push_back(Interval("x", M, N));
    b.push_back(Interval("y", P, Q));
    b.push_back(Interval("z", (1 - 1), (12-1)));
    //b.push_back(Interval("y", 0, 0));

    merge_boxes_nfm(a, b);

    std::cout << "Merged box:\n";
    for (size_t i = 0; i < a.size(); ++i) {
        std::cout << "Dim: " << a[i].var << "\n  min: " << a[i].min << "\n  max: " << a[i].max << "\n";
    }
}

void simplify_interval_test() {
    Expr x = Variable::make(Int(32), "x");
    Expr y = Variable::make(Int(32), "y");
    Expr z = Variable::make(Int(32), "z");
    Expr w = Variable::make(Int(32), "w");
    Expr M = Variable::make(Int(32), "M");
    Expr N = Variable::make(Int(32), "N");
    Expr P = Variable::make(Int(32), "P");

    vector<string> sym_consts = {"P", "N", "M"};
    vector<string> dims = {"x", "y", "z", "w"};

    Interval interval("dim", min(3, min(M, N-1)), max(3, min((N-M)/2*2+M, N-1+1)));

    std::cout << "Start simplifying interval:\n"
              << "  Min: " << interval.min << "\n"
              << "  Max: " << interval.max << "\n";

    Interval result = nfm_simplify_interval(interval);
    std::cout << "Simplified interval:\n"
              << "  Min: " << result.min << "\n"
              << "  Max: " << result.max << "\n\n";
}

void test() {
    Expr a = Variable::make(Int(32), "a");
    Expr b = Variable::make(Int(32), "b");
    Expr x = Variable::make(Int(32), "x");
    Expr y = Variable::make(Int(32), "y");
    Expr z = Variable::make(Int(32), "z");
    Expr w = Variable::make(Int(32), "w");
    Expr s = Variable::make(Int(32), "s");
    Expr t = Variable::make(Int(32), "t");
    Expr u = Variable::make(Int(32), "u");
    Expr t1 = Variable::make(Int(32), "t1");
    Expr t2 = Variable::make(Int(32), "t2");
    Expr t3 = Variable::make(Int(32), "t3");
    Expr t4 = Variable::make(Int(32), "t4");
    Expr t10 = Variable::make(Int(32), "t10");
    Expr p = Variable::make(Float(32), "p");
    Expr v = Variable::make(Bool(), "v");
    //vector<string> loop_dims = {"x", "y", "z", "s", "t", "u", "w"};
    vector<string> loop_dims = {"w"};

    //Expr expr = EQ::make(w, (((y - x) + 2)/2) - 1);
    //Expr expr = GE::make(w, max(max(x, y), z)+1);
    //Expr expr = w <= cast(Int(32), floor(x/2));
    //Expr expr = (2*w == x*6 + y + 15);
    //Expr expr = w <= max(((min(((((y - x) + 2)/2)*2), -1) + y) + 1), y);
    //Expr expr = Let::make("z", y/2, w <= z-1);
    //Expr expr = w <= max(((0 + select((x < 22), (x + 1), x)) - 1), y);
    //Expr expr = w >= min(min(y, min(x, min(z, (y + -2)))), min(x, min(z, s)));
    //Expr expr = w >= min(y, min(x, min(z, min(s, (y + -2)))));
    //Expr expr = w >= min(cast(Int(32), ceil(cast(Float(32), (x/2)))), (x/2));
    //Expr expr = w >= min(((min(x, (y + -3)) + 0) + 1), ((min(x, (y + -4)) + 1) - 1));
    //Expr expr = w >= max(x/2, x/2+1);
    //Expr expr = select(!(z > 0), min((x + -1), y-1), y) <= w;
    //Expr expr = w <= max((min(((((x - y)/4)*4) + y+1), (x + -3)) + 3), ((min(((((x - y)/4)*4) + y), (x + -3)) + 3) + 1));
    //Expr expr = w >= min(x, min(y, min((x + -1), min((x + -2), 0))));
    //Expr expr = w <= max(max((((10 - 1) - 0) + 0), max(min((min((((((y - x)/8)*8) + x) + 7), y) - 1), ((0 + 10) - 1)), 0)), max((((10 - 1) - 0) + 0), max(min((min((((((y - x)/8)*8) + x) + 7), y) + 1), ((0 + 10) - 1)), 0)));
    //Expr expr = w <= max((((10 - 1) - 0) + 0), max(min((min((((((y - x)/8)*8) + x) + 7), y) + 1), ((0 + 10) - 1)), 0));
    //Expr expr = EQ::make(w, (((((((((y - x) + 16)/16)*(((z - s) + 16)/16)) + -1)/(((y - x) + 16)/16))*16) + s) + 15));
    //Expr expr = min(select((y < x), (y + -2), (x + -3)), (min(y, (x + -1)) + -2)) <= w;

    /*Expr expr = EQ::make(w,
        max(max(max(s, u), u), max(max(max(min((((y - x)/2)*2) + x, y - 1) + 1, 3), x), u)) + 1
        - min(min(min(z, t), t), min(min(min(min(x, y-1) + 0, 3), z), t)));*/
    /*Expr expr = EQ::make(w,
        max(max(max(s, u), u), max(max(max(min((((y - x)/2)*2) + x, y - 1) + 1, 3), x), u)) + 1
        - min(min(min(z, t), t), t));*/
    /*Expr expr = EQ::make(w,
        max(max(max(s, u), u), max(max(max(min((((y - x)/2)*2) + x, y - 1) + 1, 3), x), u)));*/

    //Expr expr = EQ::make(w, max(min(x, y), z) + 1 - min(min(z, y-1), 3));


    /*TODO:
    a_copy[i].min: min(min(x, min(y, min(z, min((x + -1), min((x + -2), 0))))), min(y, min(z, s)))
    a[i].min: min(z, min(y, min((x + -2), min(s, 0))))

    a_copy[i].min: min(min(x, min(y, min((x + -2), 0))), (x - 1))
    a[i].min: min(y, min((x + -2), 0))

    a_copy[i].min: (min((min(x, (y + -3)) + 1), min(x, (y + -4)));
    a[i].min: min((y + -4), x);

    a_copy[i].max: max(max(x, max(y, max(z, max(s, max(t, max((y + -1), max((z + -1), max((s + -1), max((t + -1), max(min((y + 1), 99), max(min((z + 1), 99), max(min((s + 1), 99), max(min((t + 1), 99), 0))))))))))))), x)
    a[i].max: max(min((t + 1), 99), max(min((s + 1), 99), max(min((z + 1), 99), max(min((y + 1), 99), max(t, max(s, max(z, max(y, max(x, 0)))))))))

    INTERESTING
    a_copy[i].min: min(x, ((min((max((((y - z) + 2)/2), 1) + 1), 0)*2) + z))
    a[i].min     : min(x, z)
    min(z, min(max(((((((y - z) + 2)/2)*2) + z) + 2), (z + 4)), x))

    a_copy[i].min: min(min(((x + y) + (z*4)), ((x + s) + -3)), ((min(((z*4) + y), (s + -3)) + x) + 1))
    a[i].min     : min(((s + x) + -3), (((z*4) + y) + x))


    a_copy[i].max: max((min(((((((x - y) + 1)/8)*8) + x) + 1), (x + -6)) + 7), (x + 1))
    a[i].max     : max(min((x + 1), ((((((x - y) + 1)/8)*8) + x) + 8)), (x + 1))

    a_copy[i].max: max((((((x - y)/6)*6) + y) + 6), (min(((((((((x - y)/6)*6) + 6)/5)*5) + ((((x - y)/6)*6) + y)) + 1), (((((x - y)/6)*6) + y) + 2)) + 4))
    a[i].max     : max((((((((((x - y)/6)*6) + 6)/5)*5) + (((x - y)/6)*6)) + y) + 5), (((((x - y)/6)*6) + y) + 6))
    a_copy[i].max: max(((a + y) + 6), (min(((b + (a + y)) + 1), ((a + y) + 2)) + 4))
    a[i].max     : max(min(((y + a) + 6), (((b + y) + a) + 5)), ((y + a) + 6))
    Max: ((y + a) + 6)

    a_copy[i].min: min(x, (min((max(((y - z) + 1), 1) + 1), 0) + z))
    a[i].min     : min(z, min((max(y, z) + 2), x))

    INTERESTING
    a_copy[i].max: max(min((((max((max(a, 1) + -1), 0)*16) + y) + 15), x), z)
    a[i].max     : max(min((y + 15), x), max(min(((y + (a*16)) + -1), x), z))
    max(z, min(max((y + 15), ((y + (a*16)) + -1)), x))

    a_copy[i].min: min(x, (let t11 = min(((min(((((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + 5)/6)*3) + -4), 0)/3)*6), (((((y - z) + 3)/3)*(((s - t) + 5)/5)) + -6)) in min((((t11/max((((y - z) + 3)/3), 1))*5) + t), (s + -4))))
    a[i].min     : min(x, select((((t11 <= 0) && (((((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + -7)/6)*6) + 6) <= ((((y - z) + 3)/3)*(((s - t) + 5)/5)))) || (((((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + -7)/6) <= 0) && ((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + -5) <= (((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + -7)/6)*6))) || ((((t11 + 6) <= (((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + -7)/6)*6)) && (6 <= ((((y - z) + 3)/3)*(((s - t) + 5)/5)))) || ((1 <= ((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + -7)/6)) && (((((y - z) + 3)/3)*(((s - t) + 5)/5)) <= 5))))), (((t11/max((((y - z) + 3)/3), 1))*5) + t), (s + -4)))

    a_copy[i].max: max(x, (let t10 = min((((min(((((((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + 5)/6)*3) + -1)/4)*4), ((((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + 5)/6)*3) + -4)) + 3)/3)*6), (((((y - z) + 3)/3)*(((s - t) + 5)/5)) + -6)) in min((((((t10 + 5)/max((((y - z) + 3)/3), 1))*5) + t) + 4), s)))
    a[i].max     : max(x, min((((((t10 + 5)/max((((y - z) + 3)/3), 1))*5) + t) + 4), s))

    a_copy[i].max: select(!a, max(x, y), y)
    a[i].max     : select(((2 <= a) || ((0 <= (a*-1)) || (a == 1))), y, x)

    a_copy[i].min: select(b, (x + 10), (x + -10))
    a[i].min     : select(((2 <= b) || (0 <= (b*-1))), (x + -10), (x + 10))

    a_copy[i].min: min(min((x + -3), y), (let t12.s = (min((((((x - y)/4)*4) + y) + 3), x) - min(min(y, (x + -3)), (min(y, (x + 1)) + 1))) in (let t15 = min(min(min(y, (x + -3)), (min(y, (x + 1)) + 1)), (min((y + 3), x) + -2)) in t15)))
    a[i].min     : (let t15 = min(min(min(y, (x + -3)), min((y + 1), (x + 2))), min((y + 1), (x + -2))) in min(t15, min(y, (x + -3))))

    a_copy[i].max: max(min((((((x - y)/4)*4) + y) + 4), (x + 1)), (let t12.s = (min((((((x - y)/4)*4) + y) + 3), x) - min(min(y, (x + -3)), (min(y, (x + 1)) + 1))) in (let t14 = min((max(min(((((x - y)/4)*4) + y), (x + -3)), (min(((((x - y)/4)*4) + y), (x + 1)) + 1)) + (((t12.s + 1)/4)*4)), (min((((((x - y)/4)*4) + y) + 3), x) + -2)) in (t14 + 3))))
    a[i].max     : (let t12.s = (min((((((x - y)/4)*4) + y) + 3), x) - min(min(y, (x + -3)), min((y + 1), (x + 2)))) in (let t14 = min(max(min((((((x - y)/4)*4) + y) + (((t12.s + 1)/4)*4)), ((x + (((t12.s + 1)/4)*4)) + -3)), min(((((((x - y)/4)*4) + y) + (((t12.s + 1)/4)*4)) + 1), ((x + (((t12.s + 1)/4)*4)) + 2))), min((((((x - y)/4)*4) + y) + 1), (x + -2))) in max((t14 + 3), min((((((x - y)/4)*4) + y) + 4), (x + 1)))))

    a_copy[i].max: max(min(((x + (y*4)) + 4), (z + 1)), (let t18 = min((max((min((((y*4) + x) + -4), (z + -3)) + 5), min(((y*4) + x), (z + -3))) + ((((min((((y*4) + x) + 3), z) - min((min((((y*4) + x) + -4), (z + -3)) + 5), min(((y*4) + x), (z + -3)))) + 1)/4)*4)), (min((((y*4) + x) + 3), z) + -2)) in (t18 + 3)))
    a[i].max     : (let t18 = min(max(min(((((y*4) + x) + ((((min((((y*4) + x) + 3), z) - min((min((((y*4) + x) + -4), (z + -3)) + 5), min(((y*4) + x), (z + -3)))) + 1)/4)*4)) + 1), ((z + ((((min((((y*4) + x) + 3), z) - min((min((((y*4) + x) + -4), (z + -3)) + 5), min(((y*4) + x), (z + -3)))) + 1)/4)*4)) + 2)), min((((y*4) + x) + ((((min((((y*4) + x) + 3), z) - min((min((((y*4) + x) + -4), (z + -3)) + 5), min(((y*4) + x), (z + -3)))) + 1)/4)*4)), ((z + ((((min((((y*4) + x) + 3), z) - min((min((((y*4) + x) + -4), (z + -3)) + 5), min(((y*4) + x), (z + -3)))) + 1)/4)*4)) + -3))), min((((y*4) + x) + 1), (z + -2))) in max((t18 + 3), min((((y*4) + x) + 4), (z + 1))))

    INTERESTING
    a_copy[i].max: max(min((((((x + (y*16)) + z) + s) + (t*16)) + -15), (((x + z) + u) + -14)), ((min(((((y + t)*16) + s) + -16), (u + -15)) + (x + z)) + 1))
    a[i].max     : min((((((x + (y*16)) + z) + s) + (t*16)) + -15), (((x + z) + u) + -14))
    */

    //Expr expr = w >= min(min(x, min(y, min(z, min((x + -1), min((x + -2), 0))))), min(y, min(z, s)));
    //Expr expr = w >= min(min(x, min(y, min((x + -2), 0))), (x - 1));
    //Expr expr = w >= min(((min(x, (y + -3)) + 0) + 1), ((min(x, (y + -4)) + 1) - 1));
    //Expr expr = w <= max(x, max(y, max(z, max(s, max(t, max((y + -1), max((z + -1), max((s + -1), max((t + -1), max(min((y + 1), 99), max(min((z + 1), 99), max(min((s + 1), 99), max(min((t + 1), 99), 0)))))))))))));
    //Expr expr = w >= min(x, ((min((max((((y - z) + 2)/2), 1) + 1), 0)*2) + z));
    //Expr expr = w >= min(min(((x + y) + (z*4)), ((x + s) + -3)), ((min(((z*4) + y), (s + -3)) + x) + 1));

    //Expr expr = w <= max((min(((((((x - y) + 1)/8)*8) + x) + 1), (x + -6)) + 7), (x + 1));
    //Expr expr = w <= max(((a + y) + 6), (min(((b + (a + y)) + 1), ((a + y) + 2)) + 4));
    //Expr expr = w >= min(x, (min((max(((y - z) + 1), 1) + 1), 0) + z));
    //Expr expr = w <= max(min((((max((max(a, 1) + -1), 0)*16) + y) + 15), x), z);

    //Expr expr = w <= max(x, Let::make("t10", min((((min(((((((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + 5)/6)*3) + -1)/4)*4), ((((((((y - z) + 3)/3)*(((s - t) + 5)/5)) + 5)/6)*3) + -4)) + 3)/3)*6), (((((y - z) + 3)/3)*(((s - t) + 5)/5)) + -6)), min((((((t10 + 5)/max((((y - z) + 3)/3), 1))*5) + t) + 4), s)));
    //Expr expr = w <= max(x, Let::make("t", min(x/2, y*6), min(t*3, z+y)));

    // TODO: need a good way to transform !v
    //Expr expr = w >= select(!v, max(x, y), y);

    //Expr expr = w <= max((min((min(x, 4) + (((max(x, 4) - min(x, 4))/8)*8)), (max(x, 4) + -7)) + 7), y);
    //Expr expr = w <= (max(x,4)-min(x,4))/8*8 + 7;
    Expr expr = w <= max(min((((((x + (y*16)) + z) + s) + (t*16)) + -15), (((x + z) + u) + -14)), ((min(((((y + t)*16) + s) + -16), (u + -15)) + (x + z)) + 1));
    std::cout << "simplify: " << simplify(expr) << "\n";

    /*std::map<std::string, Expr> expr_substitutions;
    expr_substitutions.emplace("x", y + 2);
    expr_substitutions.emplace("t", x + 2);
    Expr expr = t + 1;
    std::cout << "SUBSTITUTE: " << substitute(expr_substitutions, expr) << "\n";
    std::cout << "LET: " << Let::make("x", y+2, Let::make("t", x+2, t+1)) << "\n";
    std::cout << "SIMPLIFY LET: " << simplify(Let::make("x", y+2, Let::make("t", x+2, t+1))) << "\n";*/

    Expr simplified_expr = simplify_expr(expr, loop_dims);
}

int main(int argc, const char **argv) {
    //example_expr();
    //example_interval();
    //simplify_interval_test();
    //test();
    //boxes_merge_test();
    boxes_overlap_test();
    //box_encloses_test();
    return 0;
}
