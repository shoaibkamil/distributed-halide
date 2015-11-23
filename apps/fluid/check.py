#!/usr/bin/python

import sys

keep_going = True

filea, fileb = sys.argv[1:]

def parse_values(filename):
    with open(filename, "r") as f:
        strvalues = f.read().split()
        values = map(float, strvalues)
    return values

def rel_eq(a, b):
    rel_tol = 1e-15
    abs_tol = 1e-15
    # Backported from python 3.5 math.isclose:
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def pct_diff(a, b):
    return abs(a-b)/b

def compare_values(a, b):
    max_err = -float("inf")
    for i, p in enumerate(zip(a, b)):
        if not rel_eq(p[0], p[1]):
            if not keep_going:
                return False, i, p[0], p[1]
            else:
                print "Values are not the same: value # %d, %.13f (%s) versus %.13f (%s)" % (i+1, p[0], filea, p[1], fileb)
                max_err = max(max_err, pct_diff(p[0], p[1]))
    #print "Maximum error: %f%%" % (max_err*100)
    return True, 0, 0, 0

def check(expr, msg):
    if not expr:
        print "Check failed:"
        print msg
        sys.exit(1)

if len(sys.argv) < 3:
    print "USAGE: %s <a> <b>" % sys.argv[0]
    print "Compares two lists of real numbers (separated by whitespace) for equality."
    print "  <a> : Filename of first list"
    print "  <b> : Filename of second list"
    sys.exit(1)

avalues = parse_values(filea)
bvalues = parse_values(fileb)

check(len(avalues) == len(bvalues),
      "Lengths are not the same: %d versus %d" % (len(avalues), len(bvalues)))
compare = compare_values(avalues, bvalues)
check(compare[0],
      "Values are not the same: value # %d, %.13f (%s) versus %.13f (%s)" % (compare[1]+1, compare[2], filea, compare[3], fileb))

print "Successful match."
