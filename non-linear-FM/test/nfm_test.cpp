#include <algorithm>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>
#include <vector>

#include <isl/options.h>
#include <isl/set.h>

#include "nfm_constraint.h"
#include "nfm_domain.h"
#include "nfm_isl_interface.h"
#include "nfm_polynom.h"
#include "nfm_polynom_frac.h"
#include "nfm_solver.h"
#include "nfm_space.h"

using namespace Nfm;
using namespace Nfm::Internal;

int test_parse(struct isl_ctx *ctx) {
    std::vector<std::string> param_names = {"m", "n", "p"};
    std::vector<std::string> dim_names = {"x", "y", "z"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    NfmPolyCoeff coeff1(coeff_space);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());
    coeff1 = coeff1.add(10);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.add(-10);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.add(0);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.neg();
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.add(3, {0, 1, 0}, NFM_POSITIVE);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.add(1, {1, 0, 0}, NFM_POSITIVE);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.add(4, {1, 1, 0}, NFM_POSITIVE);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.add(5, {0, 0, -1}, NFM_POSITIVE);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.sub(13, {4, 1, 3}, NFM_NEGATIVE);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.mul(-2);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.mul(3, {1, 1, 1}, NFM_NEGATIVE);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    coeff1 = coeff1.mul(2, {0, 0, 0}, NFM_POSITIVE);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    NfmPolyCoeff gcd_term = coeff1.terms_gcd();
    printf("terms gcd %s; sign: %s; is_linear? %d\n", gcd_term.to_string().c_str(),
        gcd_term.print_sign().c_str(), gcd_term.is_linear());

    coeff1 = coeff1.exquo(2, {1, 1, 1}, NFM_POSITIVE);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    int gcd_val = coeff1.content();
    printf("gcd: %d\n", gcd_val);

    coeff1 = coeff1.fdiv(5);
    printf("%s; sign: %s; is_linear? %d\n", coeff1.to_string().c_str(),
        coeff1.print_sign().c_str(), coeff1.is_linear());

    printf("\nPOLY\n");

    NfmPoly poly1(coeff_space, space);
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    poly1 = poly1.add(10);
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    poly1 = poly1.add(3, {2, 1, 0}, NFM_POSITIVE);
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    poly1 = poly1.add(1, {1, 0, 0}, NFM_POSITIVE, {0, 2, 0});
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    poly1 = poly1.add(1, {0, 1, 0}, NFM_POSITIVE, {0, 1, 0});
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    poly1 = poly1.add(1, {0, 1, 0}, NFM_POSITIVE, {0, 1, 0});
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    poly1 = poly1.add(-3, {0, 1, 0}, NFM_POSITIVE, {1, 0, 0});
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    poly1 = poly1.add(-1, {0, 1, 0}, NFM_POSITIVE, {1, 0, 1});
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    poly1 = poly1.sub(1, {0, 1, 0}, NFM_POSITIVE, {0, 1, 0});
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    poly1 = poly1.mul(4, {0, 1, 0}, NFM_POSITIVE, {0, 1, 0});
    printf("%s; sign: %s; is_linear? %d\n", poly1.to_string().c_str(),
        poly1.print_sign().c_str(), poly1.is_linear());

    gcd_val = poly1.content();
    printf("gcd: %d\n", gcd_val);

    poly1 = poly1.exquo(2);
    printf("exquo1: %s\n", poly1.to_string().c_str());

    poly1 = poly1.exquo(2, {0, 1, 0}, NFM_POSITIVE, {0, 1, 0});
    printf("exquo2: %s\n", poly1.to_string().c_str());

    NfmConstraint cst(coeff_space, space, poly1, true);
    printf("cst: %s\n", cst.to_string().c_str());
    NfmPolyCoeff constant = cst.get_constant();
    printf("%s\n", constant.to_string().c_str());
    return 0;
}

int test_dom1(struct isl_ctx *ctx) {
    /*
     6 x3  - 6 x2 + 3 x1 - 12 >= 0
    -2 x3  + 3 x2 -   x1 - 3 >= 0
       x3  - 4 x2 +   x1 + 15 >= 0
       x3  + 0 x2 -   x1 + 15 >= 0
       x3 - 2 = 0
    */
    std::vector<std::string> param_names = {"dummy"};
    std::vector<std::string> dim_names = {"x3", "x2", "x1"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{1, 0, 0}, NfmPolyCoeff(6, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(-6, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(3, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-12, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{1, 0, 0}, NfmPolyCoeff(-2, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(3, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(-1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-3, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms3 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(-4, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(15, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms4 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(-1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(15, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms5 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-2, {0}, param_names, NFM_UNKNOWN)}
    };

    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);
    NfmPoly p3(coeff_space, space, terms3);
    NfmPoly p4(coeff_space, space, terms4);
    NfmPoly p5(coeff_space, space, terms5);

    NfmConstraint c1(coeff_space, space, p1, false);
    NfmConstraint c2(coeff_space, space, p2, false);
    NfmConstraint c3(coeff_space, space, p3, false);
    NfmConstraint c4(coeff_space, space, p4, false);
    NfmConstraint c5(coeff_space, space, p5, true);

    NfmDomain dom(coeff_space, space);
    dom.add_constraint(std::move(c1));
    dom.add_constraint(std::move(c2));
    dom.add_constraint(std::move(c3));
    dom.add_constraint(std::move(c4));
    dom.add_constraint(std::move(c5));
    printf("Dom:\n%s\n", dom.to_string().c_str());

    NfmUnionDomain union_dom(coeff_space, space);
    union_dom.add_domain(std::move(dom));
    //printf("Union Dom:\n%s\n", union_dom.to_string().c_str());

    NfmUnionDomain norm = NfmSolver::nfm_union_domain_eliminate_dims(union_dom, 0, 3);
    //printf("Union Dom:\n%s\n", norm.to_string().c_str());
    for (const auto& dom : norm.get_domains()) {
        for (const auto& cst : dom.get_equalities()) {
            printf("%s\n", cst.to_string().c_str());
        }
        for (const auto& cst : dom.get_inequalities()) {
            printf("%s\n", cst.to_string().c_str());
        }
    }
    return 0;
}

int test_dom2(struct isl_ctx *ctx) {
    /*
    for i = 0 to U
        for j = to min(2i, V)
     i         >= 0
         j     >= 0
    -i     + U >= 0
    2i - j     >= 0
       - j + V >= 0
    */
    std::vector<std::string> param_names = {"U", "V"};
    std::vector<std::string> dim_names = {"i", "j"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    // Context U - 1 >= 0
    NfmPolyCoeff coeff1(1, {1, 0}, param_names, NFM_UNKNOWN);
    coeff1 = coeff1 - 1;
    NfmContext ctx1(coeff1, false);

    // Context V - 1 >= 0
    NfmPolyCoeff coeff2(1, {0, 1}, param_names, NFM_UNKNOWN);
    coeff2 = coeff2 - 1;
    NfmContext ctx2(coeff2, false);

    NfmContextDomain ctx_dom(coeff_space);
    ctx_dom.add_context(ctx1);
    ctx_dom.add_context(ctx2);
    printf("Before simplify content:\n%s\n", ctx_dom.to_string().c_str());
    ctx_dom.simplify();
    printf("After simplify content:\n%s\n\n", ctx_dom.to_string().c_str());

    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{1, 0}, NfmPolyCoeff(1, {0, 0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{0, 1}, NfmPolyCoeff(1, {0, 0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms3 = {
        {{1, 0}, NfmPolyCoeff(-1, {0, 0}, param_names, NFM_UNKNOWN)},
        {{0, 0}, NfmPolyCoeff(1, {1, 0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms4 = {
        {{1, 0}, NfmPolyCoeff(2, {0, 0}, param_names, NFM_UNKNOWN)},
        {{0, 1}, NfmPolyCoeff(-1, {0, 0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms5 = {
        {{0, 1}, NfmPolyCoeff(-1, {0, 0}, param_names, NFM_UNKNOWN)},
        {{0, 0}, NfmPolyCoeff(1, {0, 1}, param_names, NFM_UNKNOWN)}
    };


    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);
    NfmPoly p3(coeff_space, space, terms3);
    NfmPoly p4(coeff_space, space, terms4);
    NfmPoly p5(coeff_space, space, terms5);

    NfmConstraint c1(coeff_space, space, p1, false);
    NfmConstraint c2(coeff_space, space, p2, false);
    NfmConstraint c3(coeff_space, space, p3, false);
    NfmConstraint c4(coeff_space, space, p4, false);
    NfmConstraint c5(coeff_space, space, p5, false);

    NfmDomain dom(coeff_space, space, ctx_dom);
    dom.add_constraint(std::move(c1));
    dom.add_constraint(std::move(c2));
    dom.add_constraint(std::move(c3));
    dom.add_constraint(std::move(c4));
    dom.add_constraint(std::move(c5));
    printf("Dom:\n%s\n", dom.to_string().c_str());
    printf("Context: %s\n\n", dom.get_context_domain().to_string().c_str());

    NfmUnionDomain union_dom(coeff_space, space);
    union_dom.add_domain(std::move(dom));
    printf("Union Dom:\n%s\n", union_dom.to_string().c_str());

    NfmUnionDomain norm = NfmSolver::nfm_union_domain_eliminate_dims(union_dom, 0, 2);
    //printf("Union Dom:\n%s\n", norm.to_string().c_str());
    for (const auto& dom : norm.get_domains()) {
        for (const auto& cst : dom.get_equalities()) {
            printf("%s\n", cst.to_string().c_str());
        }
        for (const auto& cst : dom.get_inequalities()) {
            printf("%s\n", cst.to_string().c_str());
        }
    }
    return 0;
}

int test_dom3(struct isl_ctx *ctx) {
    /*
    for i = 0 to U
      for j = 0 to min(2i, V)

     i         >= 0
         j     >= 0
    -i     + U >= 0
    2i - j     >= 0
       - j + V >= 0

    Tiling original loop on j:
    -bx + j           >= 0
     bx - j + (b - 1) >= 0
    */
    std::vector<std::string> param_names = {"U", "V", "b"};
    std::vector<std::string> dim_names = {"x", "i", "j"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    // Context U - 1 >= 0
    NfmPolyCoeff coeff1(1, {1, 0, 0}, param_names, NFM_UNKNOWN);
    coeff1 = coeff1 - 1;
    NfmContext ctx1(coeff1, false);

    // Context V - 1 >= 0
    NfmPolyCoeff coeff2(1, {0, 1, 0}, param_names, NFM_UNKNOWN);
    coeff2 = coeff2 - 1;
    NfmContext ctx2(coeff2, false);

    // Context b - 2 >= 0
    NfmPolyCoeff coeff3(1, {0, 0, 1}, param_names, NFM_UNKNOWN);
    coeff3 = coeff3 - 2;
    NfmContext ctx3(coeff3, false);

    NfmContextDomain ctx_dom(coeff_space);
    ctx_dom.add_context(ctx1);
    ctx_dom.add_context(ctx2);
    ctx_dom.add_context(ctx3);
    printf("\nBefore simplify content:\n%s\n", ctx_dom.to_string().c_str());
    ctx_dom.simplify();
    printf("After simplify content:\n%s\n\n", ctx_dom.to_string().c_str());

    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{0, 1, 0}, NfmPolyCoeff(1, {0, 0, 0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{0, 0, 1}, NfmPolyCoeff(1, {0, 0, 0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms3 = {
        {{0, 1, 0}, NfmPolyCoeff(-1, {0, 0, 0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(1, {1, 0, 0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms4 = {
        {{0, 1, 0}, NfmPolyCoeff(2, {0, 0, 0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(-1, {0, 0, 0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms5 = {
        {{0, 0, 1}, NfmPolyCoeff(-1, {0, 0, 0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(1, {0, 1, 0}, param_names, NFM_UNKNOWN)}
    };
    // -bx + j >= 0
    std::map<std::vector<int>, NfmPolyCoeff> terms6 = {
        {{1, 0, 0}, NfmPolyCoeff(-1, {0, 0, 1}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(1, {0, 0, 0}, param_names, NFM_UNKNOWN)}
    };
    // bx - j + (b - 1) >= 0
    NfmPolyCoeff coeff = NfmPolyCoeff(1, {0, 0, 1}, param_names, NFM_UNKNOWN);
    coeff = coeff - 1;
    std::map<std::vector<int>, NfmPolyCoeff> terms7 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0, 0, 1}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(-1, {0, 0, 0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, coeff}
    };

    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);
    NfmPoly p3(coeff_space, space, terms3);
    NfmPoly p4(coeff_space, space, terms4);
    NfmPoly p5(coeff_space, space, terms5);
    NfmPoly p6(coeff_space, space, terms6);
    NfmPoly p7(coeff_space, space, terms7);

    NfmConstraint c1(coeff_space, space, p1, false);
    NfmConstraint c2(coeff_space, space, p2, false);
    NfmConstraint c3(coeff_space, space, p3, false);
    NfmConstraint c4(coeff_space, space, p4, false);
    NfmConstraint c5(coeff_space, space, p5, false);
    NfmConstraint c6(coeff_space, space, p6, false);
    NfmConstraint c7(coeff_space, space, p7, false);

    NfmDomain dom(coeff_space, space, ctx_dom);
    dom.add_constraint(std::move(c1));
    dom.add_constraint(std::move(c2));
    dom.add_constraint(std::move(c3));
    dom.add_constraint(std::move(c4));
    dom.add_constraint(std::move(c5));
    dom.add_constraint(std::move(c6));
    dom.add_constraint(std::move(c7));
    printf("Dom:\n%s\n\n", dom.to_string_with_sign().c_str());

    NfmUnionDomain union_dom(coeff_space, space);
    union_dom.add_domain(std::move(dom));
    //printf("Union Dom:\n%s\n", union_dom.to_string().c_str());

    NfmUnionDomain norm = NfmSolver::nfm_union_domain_eliminate_dims(union_dom, 0, 3);
    //printf("Union Dom:\n%s\n", norm.to_string().c_str());
    /*for (const auto& dom : norm.get_domains()) {
        for (const auto& cst : dom.get_equalities()) {
            printf("%s\n", cst.to_string().c_str());
        }
        for (const auto& cst : dom.get_inequalities()) {
            printf("%s\n", cst.to_string().c_str());
        }
    }*/
    return 0;
}

int test_dom4(struct isl_ctx *ctx) {
    std::vector<std::string> param_names = {"a", "b", "c"};
    std::vector<std::string> dim_names = {"x", "y"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    // ax + by >= 0
    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{1, 0}, NfmPolyCoeff(1, {1, 0, 0}, param_names, NFM_UNKNOWN)},
        {{0, 1}, NfmPolyCoeff(1, {0, 1, 0}, param_names, NFM_UNKNOWN)}
    };
    // cx + (b - 1) >= 0
    NfmPolyCoeff coeff = NfmPolyCoeff(1, {0, 1, 0}, param_names, NFM_UNKNOWN);
    coeff = coeff - 1;
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{1, 0}, NfmPolyCoeff(1, {0, 0, 1}, param_names, NFM_UNKNOWN)},
        {{0, 0}, coeff}
    };

    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);

    NfmConstraint c1(coeff_space, space, p1, false);
    NfmConstraint c2(coeff_space, space, p2, false);

    NfmDomain dom(coeff_space, space);
    dom.add_constraint(std::move(c1));
    dom.add_constraint(std::move(c2));
    printf("Dom:\n%s\n\n", dom.to_string_with_sign().c_str());

    NfmUnionDomain union_dom(coeff_space, space);
    union_dom.add_domain(std::move(dom));
    //printf("Union Dom:\n%s\n", union_dom.to_string().c_str());

    NfmUnionDomain norm = NfmSolver::nfm_union_domain_eliminate_dims(union_dom, 0, 2);
    //printf("Union Dom:\n%s\n", norm.to_string().c_str());
    printf("Size: %d\n", (int)norm.get_num_domains());
    for (size_t i = 0; i < norm.get_num_domains(); ++i) {
        printf("\nDomain %d\n", (int)i);
        const auto& dom = norm[i];
        for (const auto& cst : dom.get_equalities()) {
            printf("  %s\n", cst.to_string_with_sign().c_str());
        }
        for (const auto& cst : dom.get_inequalities()) {
            printf("  %s\n", cst.to_string_with_sign().c_str());
        }
        printf("Context: %s\n\n", dom.get_context_domain().to_string().c_str());
    }
    return 0;
}

int test_classify_unknown(struct isl_ctx *ctx) {
    std::vector<std::string> param_names = {"a", "b", "c", "d"};
    std::vector<std::string> dim_names = {"x"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    // Context b > 0 (b - 1 >=0)
    NfmPolyCoeff coeff1(1, {0, 1, 0, 0}, param_names, NFM_UNKNOWN);
    coeff1 = coeff1 - 1;
    NfmContext ctx1(coeff1, false);

    // Context a - b == 4
    NfmPolyCoeff coeff2(1, {1, 0, 0, 0}, param_names, NFM_UNKNOWN);
    coeff2 = coeff2 - coeff1 - 5;
    NfmContext ctx2(coeff2, true);

    // Context 3*c*d >= 3
    NfmPolyCoeff coeff3(3, {0, 0, 1, 1}, param_names, NFM_UNKNOWN);
    coeff3 = coeff3 - 3;
    NfmContext ctx3(coeff3, true);

    NfmContextDomain ctx_dom(coeff_space);
    ctx_dom.add_context(ctx1);
    ctx_dom.add_context(ctx2);
    ctx_dom.add_context(ctx3);

    printf("Before simplify content:\n%s\n", ctx_dom.to_string().c_str());
    ctx_dom.simplify();
    printf("After simplify content:\n%s\n\n", ctx_dom.to_string().c_str());

    // ax >= 0
    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{1}, NfmPolyCoeff(1, {1, 0, 0, 0}, param_names, NFM_UNKNOWN)},
    };
    // bx >= 0
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{1}, NfmPolyCoeff(1, {0, 1, 0, 0}, param_names, NFM_UNKNOWN)},
    };
    // cx >= 0
    std::map<std::vector<int>, NfmPolyCoeff> terms3 = {
        {{1}, NfmPolyCoeff(1, {0, 0, 1, 0}, param_names, NFM_UNKNOWN)},
    };
    // dx >= 0
    std::map<std::vector<int>, NfmPolyCoeff> terms4 = {
        {{1}, NfmPolyCoeff(1, {0, 0, 0, 1}, param_names, NFM_UNKNOWN)},
    };

    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);
    NfmPoly p3(coeff_space, space, terms3);
    NfmPoly p4(coeff_space, space, terms4);

    NfmConstraint c1(coeff_space, space, p1, false);
    NfmConstraint c2(coeff_space, space, p2, false);
    NfmConstraint c3(coeff_space, space, p3, false);
    NfmConstraint c4(coeff_space, space, p4, false);

    NfmDomain dom(coeff_space, space, ctx_dom);
    dom.add_constraint(std::move(c1));
    dom.add_constraint(std::move(c2));
    dom.add_constraint(std::move(c3));
    dom.add_constraint(std::move(c4));
    printf("Dom:\n%s\n", dom.to_string_with_sign().c_str());
    printf("Context: %s\n\n", dom.get_context_domain().to_string().c_str());

    NfmUnionDomain result = NfmSolver::nfm_domain_classify_unknown_coefficient(dom, 0);
    printf("Size: %d\n", (int)result.get_domains().size());
    for (size_t i = 0; i < result.get_domains().size(); ++i) {
        const auto& domain = result[i];
        printf("Domain %d:\n%s\n", (int)i, domain.to_string_with_sign().c_str());
        printf("Context: %s\n\n", domain.get_context_domain().to_string().c_str());
    }
    return 0;
}

int test_dom_equal(struct isl_ctx *ctx) {
    /*
     6 x3  - 6 x2 + 3 x1 - 12 >= 0
    -2 x3  + 3 x2 -   x1 - 3 >= 0
       x3  - 4 x2 +   x1 + 15 >= 0
       x3  + 0 x2 -   x1 + 15 >= 0
       x3 - 2 = 0
    */
    std::vector<std::string> param_names = {"dummy"};
    std::vector<std::string> dim_names = {"x3", "x2", "x1"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{1, 0, 0}, NfmPolyCoeff(6, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(-6, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(3, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-12, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{1, 0, 0}, NfmPolyCoeff(-2, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(3, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(-1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-3, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms3 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(-4, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(15, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms4 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(-1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(15, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms5 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-2, {0}, param_names, NFM_UNKNOWN)}
    };

    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);
    NfmPoly p3(coeff_space, space, terms3);
    NfmPoly p4(coeff_space, space, terms4);
    NfmPoly p5(coeff_space, space, terms5);

    NfmConstraint c1(coeff_space, space, p1, false);
    NfmConstraint c2(coeff_space, space, p2, false);
    NfmConstraint c3(coeff_space, space, p3, false);
    NfmConstraint c4(coeff_space, space, p4, false);
    NfmConstraint c5(coeff_space, space, p5, true);

    NfmDomain dom1(coeff_space, space);
    dom1.add_constraint(c1);
    dom1.add_constraint(c2);
    dom1.add_constraint(c3);
    dom1.add_constraint(c4);
    dom1.add_constraint(c5);

    NfmDomain dom2(coeff_space, space);
    dom2.add_constraint(c3);
    dom2.add_constraint(c2);
    dom2.add_constraint(c5);
    dom2.add_constraint(c4);
    dom2.add_constraint(c1);
    bool is_equal = (dom1 == dom2);

    printf("Dom1:\n%s\n", dom1.to_string().c_str());
    printf("Dom2:\n%s\n", dom2.to_string().c_str());
    printf("Dom1 == Dom2? %d\n", is_equal);
    assert(is_equal);

    NfmUnionDomain udom(coeff_space, space);
    udom.add_domain(dom1);
    udom.add_domain(dom1);
    printf("Union Domain: %s\n", udom.to_string().c_str());
    assert(udom.get_domains().size() == 1);
    return 0;
}

int test_poly_term(struct isl_ctx *ctx) {
    std::vector<std::string> param_names = {"M", "N", "P"};
    NfmSpace coeff_space(param_names);

    // (N - 1) >= 0
    std::map<std::vector<int>, int> terms = {
        {{2, 1, 0}, 2},
        {{0, 0, 1}, 3}
    };
    NfmPolyCoeff p1 = NfmPolyCoeff(terms, param_names, NFM_UNKNOWN);

    auto res = p1.get_coeff_involving_dim(0);
    printf("Term: %s\n", res.first.to_string().c_str());
    printf("Coeff: %s\n", res.second.to_string().c_str());
    return 0;
}

int test_poly_compare(struct isl_ctx *ctx) {
    /*
       x3  - 6 x2 + 3 x1 - 12 >= 0
       x3  + 3 x2 -   x1 - 3 >= 0
       - 4 x2 +   x1 + 15 >= 0
       x3  + 0 x2 -   x1 + 15 >= 0
       x3 - 2 = 0
    */
    std::vector<std::string> param_names = {"dummy"};
    std::vector<std::string> dim_names = {"x3", "x2", "x1"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(-6, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(3, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-12, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{2, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(3, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(-1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-3, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms3 = {
        {{0, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(-4, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(15, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms4 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(-1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(15, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms5 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-2, {0}, param_names, NFM_UNKNOWN)}
    };

    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);
    NfmPoly p3(coeff_space, space, terms3);
    NfmPoly p4(coeff_space, space, terms4);
    NfmPoly p5(coeff_space, space, terms5);

    std::vector<NfmPoly> v = {p1, p2, p1, p3, p4, p5};
    printf("Before Sort NfmPoly\n");
    for (auto& poly : v) {
        printf("  %s\n", poly.to_string().c_str());
    }
    std::sort(v.begin(), v.end());
    printf("\nAfter Sort NfmPoly\n");
    for (auto& poly : v) {
        printf("  %s\n", poly.to_string().c_str());
    }

    NfmConstraint c1(coeff_space, space, p1, true);
    NfmConstraint c2(coeff_space, space, p2, false);
    NfmConstraint c3(coeff_space, space, p3, false);
    NfmConstraint c4(coeff_space, space, p4, false);
    NfmConstraint c5(coeff_space, space, p5, true);
    std::vector<NfmConstraint> csts = {c1, c3, c2, c4, c5, c2, c3};
    printf("\nBefore Sort NfmConstraint\n");
    for (auto& cst : csts) {
        printf("  %s\n", cst.to_string().c_str());
    }
    std::sort(csts.begin(), csts.end());
    printf("\nAfter Sort NfmConstraint\n");
    for (auto& cst : csts) {
        printf("  %s\n", cst.to_string().c_str());
    }

    NfmDomain dom1(coeff_space, space);
    dom1.add_constraint(c1);
    dom1.add_constraint(c2);
    dom1.add_constraint(c3);

    NfmDomain dom2(coeff_space, space);
    dom2.add_constraint(c3);
    dom2.add_constraint(c4);
    dom2.add_constraint(c5);

    NfmUnionDomain udom(coeff_space, space);
    udom.add_domain(dom1);
    udom.add_domain(dom2);

    printf("\nBefore Sort NfmUnionDomain\n");
    for (auto& dom : udom.get_domains()) {
        printf("Domain: \n");
        for (auto& cst : dom.get_constraints()) {
            printf("  %s\n", cst.to_string().c_str());
        }
    }
    udom.sort();
    printf("\nAfter Sort NfmUnionDomain\n");
    for (auto& dom : udom.get_domains()) {
        printf("Domain: \n");
        for (auto& cst : dom.get_constraints()) {
            printf("  %s\n", cst.to_string().c_str());
        }
    }
    return 0;
}

int test_redundant(struct isl_ctx *ctx) {
    /*
             w - 3  = 0
    -y         + 2 >= 0
     y - u         >= 0
       - u     + 2 >= 0 (Redundant constraint)
    */
    std::vector<std::string> param_names = {"dummy"};
    std::vector<std::string> dim_names = {"y", "u", "w"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{1, 0, 0}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(-3, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{1, 0, 0}, NfmPolyCoeff(-1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(2, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms3 = {
        {{1, 0, 0}, NfmPolyCoeff(1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(-1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms4 = {
        {{1, 0, 0}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 1, 0}, NfmPolyCoeff(-1, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 1}, NfmPolyCoeff(0, {0}, param_names, NFM_UNKNOWN)},
        {{0, 0, 0}, NfmPolyCoeff(2, {0}, param_names, NFM_UNKNOWN)}
    };

    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);
    NfmPoly p3(coeff_space, space, terms3);
    NfmPoly p4(coeff_space, space, terms4);

    NfmConstraint c1(coeff_space, space, p1, true);
    NfmConstraint c2(coeff_space, space, p2, false);
    NfmConstraint c3(coeff_space, space, p3, false);
    NfmConstraint c4(coeff_space, space, p4, false);

    NfmDomain dom(coeff_space, space);
    dom.add_constraint(std::move(c1));
    dom.add_constraint(std::move(c2));
    dom.add_constraint(std::move(c3));
    dom.add_constraint(std::move(c4));

    NfmUnionDomain union_dom(coeff_space, space);
    union_dom.add_domain(std::move(dom));

    NfmUnionDomain simplified = NfmSolver::nfm_union_domain_simplify(union_dom);
    printf("Is empty: %d\n", simplified.is_empty());
    for (const auto& dom : simplified.get_domains()) {
        for (const auto& cst : dom.get_equalities()) {
            printf("%s\n", cst.to_string().c_str());
        }
        for (const auto& cst : dom.get_inequalities()) {
            printf("%s\n", cst.to_string().c_str());
        }
    }

    bool is_redundant = NfmSolver::nfm_constraint_ineq_is_redundant(union_dom, c4);
    printf("Constraint %s is redundant? %d\n", c4.to_string().c_str(), is_redundant);
    return 0;
}

int test_correctness(struct isl_ctx *ctx) {
    /*
       -a - x - y + 16 >= 0
       -a + x + y      >= 0
        a - x - y      >= 0
        a + x + y - 16 >= 0
       -b + 8 >= 0
        b - 8 >= 0
    */
    std::vector<std::string> param_names = {"x", "y"};
    std::vector<std::string> dim_names = {"a", "b"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{1, 0}, -NfmPolyCoeff::make_one(coeff_space)},
        {{0, 0}, NfmPolyCoeff({{{1, 0}, -1}, {{0, 1}, -1}, {{0, 0}, 16}}, param_names, NFM_UNKNOWN)},
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{1, 0}, -NfmPolyCoeff::make_one(coeff_space)},
        {{0, 0}, NfmPolyCoeff({{{1, 0}, 1}, {{0, 1}, 1}}, param_names, NFM_UNKNOWN)},
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms3 = {
        {{1, 0}, NfmPolyCoeff::make_one(coeff_space)},
        {{0, 0}, NfmPolyCoeff({{{1, 0}, -1}, {{0, 1}, -1}}, param_names, NFM_UNKNOWN)},
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms4 = {
        {{1, 0}, NfmPolyCoeff::make_one(coeff_space)},
        {{0, 0}, NfmPolyCoeff({{{1, 0}, 1}, {{0, 1}, 1}, {{0, 0}, -16}}, param_names, NFM_UNKNOWN)},
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms5 = {
        {{0, 1}, -NfmPolyCoeff::make_one(coeff_space)},
        {{0, 0}, NfmPolyCoeff(8, {0, 0}, param_names, NFM_UNKNOWN)}
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms6 = {
        {{0, 1}, NfmPolyCoeff::make_one(coeff_space)},
        {{0, 0}, NfmPolyCoeff(-8, {0, 0}, param_names, NFM_UNKNOWN)}
    };

    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);
    NfmPoly p3(coeff_space, space, terms3);
    NfmPoly p4(coeff_space, space, terms4);
    NfmPoly p5(coeff_space, space, terms5);
    NfmPoly p6(coeff_space, space, terms6);

    NfmConstraint c1(coeff_space, space, p1, false);
    NfmConstraint c2(coeff_space, space, p2, false);
    NfmConstraint c3(coeff_space, space, p3, false);
    NfmConstraint c4(coeff_space, space, p4, false);
    NfmConstraint c5(coeff_space, space, p5, false);
    NfmConstraint c6(coeff_space, space, p6, false);

    NfmDomain dom1(coeff_space, space);
    dom1.add_constraint(c1);
    dom1.add_constraint(c2);
    dom1.add_constraint(c3);
    dom1.add_constraint(c4);
    dom1.add_constraint(c5);
    dom1.add_constraint(c6);

    NfmUnionDomain udom(coeff_space, space);
    udom.add_domain(dom1);

    printf("\nBefore simplification\n");
    for (auto& dom : udom.get_domains()) {
        printf("Domain: \n");
        for (auto& cst : dom.get_constraints()) {
            printf("  %s\n", cst.to_string().c_str());
        }
    }

    NfmUnionDomain norm = NfmSolver::nfm_union_domain_simplify(udom);
    norm.sort();
    printf("\nAfter simplification NFM\n");
    for (auto& dom : norm.get_domains()) {
        printf("Domain: \n");
        for (auto& cst : dom.get_constraints()) {
            printf("  %s\n", cst.to_string().c_str());
        }
        printf("Context: \n  %s\n", dom.get_context_domain().to_string().c_str());
    }

    NfmUnionDomain simplified_isl = udom.simplify();
    printf("\nAfter simplification ISL\n");
    for (auto& dom : simplified_isl.get_domains()) {
        printf("Domain: \n");
        for (auto& cst : dom.get_constraints()) {
            printf("  %s\n", cst.to_string().c_str());
        }
    }
    return 0;
}

int test_correctness2(struct isl_ctx *ctx) {
    /*
        a - x - y      = 0
        x + y - 8 = 0
    */
    std::vector<std::string> param_names = {"x", "y"};
    std::vector<std::string> dim_names = {"a", "b"};
    NfmSpace coeff_space(param_names);
    NfmSpace space(dim_names);

    std::map<std::vector<int>, NfmPolyCoeff> terms1 = {
        {{1, 0}, -NfmPolyCoeff::make_one(coeff_space)},
        {{0, 0}, NfmPolyCoeff({{{1, 0}, -1}, {{0, 1}, -1}, {{0, 0}, 16}}, param_names, NFM_UNKNOWN)},
    };
    std::map<std::vector<int>, NfmPolyCoeff> terms2 = {
        {{0, 0}, NfmPolyCoeff({{{1, 0}, 1}, {{0, 1}, 1}, {{0, 0}, -8}}, param_names, NFM_UNKNOWN)},
    };

    NfmPoly p1(coeff_space, space, terms1);
    NfmPoly p2(coeff_space, space, terms2);

    NfmConstraint c1(coeff_space, space, p1, true);
    NfmConstraint c2(coeff_space, space, p2, true);

    NfmDomain dom1(coeff_space, space);
    dom1.add_constraint(c1);
    dom1.add_constraint(c2);

    NfmUnionDomain udom(coeff_space, space);
    udom.add_domain(dom1);

    printf("\nBefore simplification\n");
    for (auto& dom : udom.get_domains()) {
        printf("Domain: \n");
        for (auto& cst : dom.get_constraints()) {
            printf("  %s\n", cst.to_string().c_str());
        }
    }

    NfmUnionDomain norm = NfmSolver::nfm_union_domain_simplify(udom);
    norm.sort();
    printf("\nAfter simplification NFM\n");
    for (auto& dom : norm.get_domains()) {
        printf("Domain: \n");
        for (auto& cst : dom.get_constraints()) {
            printf("  %s\n", cst.to_string().c_str());
        }
        printf("Context: \n  %s\n", dom.get_context_domain().to_string().c_str());
    }

    NfmUnionDomain simplified_isl = udom.simplify();
    printf("\nAfter simplification ISL\n");
    for (auto& dom : simplified_isl.get_domains()) {
        printf("Domain: \n");
        for (auto& cst : dom.get_constraints()) {
            printf("  %s\n", cst.to_string().c_str());
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    struct isl_ctx *ctx;
    struct isl_options *options;

    options = isl_options_new_with_defaults();
    argc = isl_options_parse(options, argc, argv, ISL_ARG_ALL);
    ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

    /*std::string str = "[M, N] -> {[x, y, z, w] : (-1 + -1*y + 3*x >= 0 and z >= 0 and -2 + 2*w + -1*z + 3*x >= 0) "
        "or (y + -3*x >= 0 and z >= 0 and -1 + 2*w + -1*z + y >= 0) "
        "or (-1 + -1*z + -1*y + 3*x >= 0 and -1 + -1*z >= 0 and -2 + 2*w + -1*z + 3*x >= 0) "
        "or (z + y + -3*x >= 0 and -1 + -1*z >= 0 and -1 + 2*w + y >= 0)}";*/
    //std::string str = "[M, N] -> {[x, y, z, w, s, u] : w = max(u, max(max(y, 3), u))}";
    /*std::string str = "[] -> {[x, y, z, s, t, u, w] : w = "
        "max(max(max(s, u), u), max(max(max(min((((y - x)/2)*2) + x, y - 1) + 1, 3), x), u)) + 1"
        "- min(min(min(z, t), t), min(min(min(min(x, y-1) + 0, 3), z), t))}";*/

    std::string str = "[x, y] -> {[a, b] : a + x + y + -16 = 0 and x+y-8 = 0}";
    isl_set *set = isl_set_read_from_str(ctx, str.c_str());
    isl_set_dump(set);
    //printf("set size: %d\n", isl_set_n_basic_set(set));
    //printf("is universe? %d\n", isl_set_plain_is_universe(set));
    //printf("is empty? %d\n", isl_set_plain_is_empty(set));
    /*isl_bset_list *bset_list = isl_set_get_bsets_list(set);
    isl_bset_list_dump(bset_list);
    isl_bset_list_free(bset_list);

    printf("\n");
    isl_set *disjoint = isl_set_make_disjoint(set);
    bset_list = isl_set_get_bsets_list(disjoint);
    isl_bset_list_dump(bset_list);
    isl_bset_list_free(bset_list);
    isl_set_free(disjoint);*/

    isl_set_free(set);

    //test_parse(ctx);
    //test_classify_unknown(ctx);
    //test_dom1(ctx);
    //test_dom2(ctx);
    //test_dom3(ctx);
    //test_dom4(ctx);
    //test_dom_equal(ctx);
    //test_poly_term(ctx);
    //test_poly_compare(ctx);
    //test_redundant(ctx);
    //test_correctness2(ctx);
    isl_ctx_free(ctx);

    return 0;
}
