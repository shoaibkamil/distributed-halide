#!/usr/bin/python

import numpy as np
from optparse import OptionParser
import os
import re
import subprocess
import sys

class Config:
    def __init__(self, output_dir, nodes, srcfile, baseline, args):
        self.output_dir = output_dir
        self.nodes = map(int, nodes.split(","))
        self.tasks_per_node = 2
        self.cores_per_socket = 12
        self.srcfile = srcfile
        self.baseline = float(baseline) if baseline else None
        self.exe = args
        self.input_size = "%sx%s" % (args[-2], args[-1])

class Timing:
    def __init__(self, runtime, percentile20, percentile80):
        self.runtime = runtime
        self.percentile20 = percentile20
        self.percentile80 = percentile80

class Result:
    def __init__(self, cmd, num_nodes, value):
        self.cmd = cmd
        self.num_nodes = num_nodes
        self.timing = value

def make_run_cmd(config, num_nodes, baseline=False):
    os.environ["MV2_ENABLE_AFFINITY"] = "0"
    if baseline:
        os.environ["HL_DISABLE_DISTRIBUTED"] = "1"
    else:
        os.environ["HL_DISABLE_DISTRIBUTED"] = "0"
    cmd = ["srun", "--exclusive"]
    # if nranks == num_nodes * 2:
    #     cmd.extend(["--cpu_bind=verbose,sockets"])
    # elif nranks > num_nodes:
    #     cmd.extend(["--cpu_bind=verbose,cores"])
    # else:
    #     cmd.extend(["--cpu_bind=verbose"])
    if baseline:
        nranks = num_nodes
        cmd.extend(["--cpu_bind=verbose"])
    else:
        nranks = num_nodes * config.tasks_per_node
        cmd.extend(["--cpu_bind=verbose,sockets"])
    cmd.extend(["--ntasks=%d" % nranks])
    cmd.extend(["--nodes=%d" % num_nodes])
    cmd.extend(config.exe)
    cmd = map(str, cmd)
    return cmd

def execute(cmd):
    print "Executing HL_DISABLE_DISTRIBUTED=%s %s..." % (os.environ["HL_DISABLE_DISTRIBUTED"], " ".join(cmd))
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print e
        return ""

def readall(path):
    with open(path, "r") as f:
        contents = f.read()
    return contents

def get_time(output):
    # Timing: <16> ranks <0.000128> seconds, 20/80 percentile <0.000070,0.000240>
    exp = "Timing: <\d+> ranks <(\d*?\.\d+?)> seconds, 20/80 percentile <(\d*?\.\d+?),(\d*?\.\d+?)>"
    m = re.search(exp, output)
    if m:
        runtime = float(m.groups()[0])
        percentile20 = float(m.groups()[1])
        percentile80 = float(m.groups()[2])
        return Timing(runtime, percentile20, percentile80)
    else:
        return -1

def get_unique_path(template):
    i = 0
    while True:
        path = template % i
        i += 1
        if not os.path.exists(path):
            return path
    sys.exit(1)

def run(config):
    results = {}
    if not config.baseline:
        # Run baseline of a single node
        cmd = make_run_cmd(config, 1, baseline=True)
        result = get_time(execute(cmd))
        print "Result was: %f sec" % result.runtime
        results["baseline"] = Result(["HL_DISABLE_DISTRIBUTED=1"] + cmd, 1, result)
    else:
        results["baseline"] = Result("Baseline specified on command line", 1, config.baseline)
    # Run all other tests.
    for num_nodes in config.nodes:
        nranks = num_nodes * config.tasks_per_node if num_nodes > 1 else 1
        cmd = make_run_cmd(config, num_nodes)
        result = get_time(execute(cmd))
        print "Result was: %f sec" % result.runtime
        results[nranks] = Result(cmd, num_nodes, result)
    return results

def calc_speedup(singlerank, numranks, timing):
    try:
        rt = singlerank / timing.runtime
        # Swap the percentiles because we're calculating percentiles
        # of speedup, not runtime.
        pct20 = singlerank / (timing.percentile80)
        pct80 = singlerank / (timing.percentile20)
        return Timing(rt, pct20, pct80)
    except ZeroDivisionError:
        return Timing(numranks, numranks, numranks)

def calculate_highest_speedup(config, results):
    ranks = max(filter(lambda x: isinstance(x, int), results.keys()))
    singlerank = results["baseline"].timing.runtime
    speedup = calc_speedup(singlerank, ranks, results[ranks].timing)
    linear_speedup = (ranks * config.cores_per_socket) / float(2 * config.cores_per_socket)
    pct_of_linear = (float(speedup.runtime) / linear_speedup) * 100
    return (ranks, speedup.runtime, pct_of_linear)

def report(config, results):
    best = calculate_highest_speedup(config, results)
    print "Baseline (%d cores) runtime: %f sec" % (2 * config.cores_per_socket, results["baseline"].timing.runtime)
    print "%d rank (%d cores) runtime: %f" % (best[0], best[0] * config.cores_per_socket, results[best[0]].timing.runtime)
    print "%d rank speedup: %.3f (%.1f%% of linear)" % (best[0], best[1], best[2])
    if config.output_dir == None:
        return
    contents = ""
    contents += config.exe[0] + "\n"
    contents += "Input size: %s\n" % config.input_size
    contents += "Testing node counts %s\n" % ", ".join(map(str, config.nodes))
    contents += "Binding to sockets, meaning 1 rank = %d cores\n" % config.cores_per_socket
    contents += "Invocation commands:\n"
    for k, v in results.items():
        nranks = str(k)
        contents += " %s: %s\n" % (nranks, " ".join(v.cmd))
    contents += "Runtime and speedup per rank count (in seconds):\n"
    contents += "--BEGIN DATA--\n"
    singlerank = results["baseline"].timing.runtime
    for k, v in results.items():
        nranks = str(k)
        speedup = calc_speedup(singlerank, nranks, v.timing)
        # speedup.runtime is actually a speedup, not a runtime.
        contents += "%s: %f\t%.3f\n" % (nranks, v.timing.runtime, speedup.runtime)
    contents += "--END DATA--\n"
    contents += "Raw source dump:\n"
    contents += "// File name %s\n" % config.srcfile
    contents += readall(config.srcfile)

    exe = os.path.basename(config.exe[0])
    txttemplate = "%s-%s-%s-%%d.txt" % (exe, ",".join(map(str, config.nodes)),
                                        config.input_size)
    dattemplate = "%s-%s-%s-%%d.dat" % (exe, ",".join(map(str, config.nodes)),
                                        config.input_size)
    txtpath = get_unique_path(os.path.join(config.output_dir, txttemplate))
    datpath = get_unique_path(os.path.join(config.output_dir, dattemplate))
    with open(txtpath, "w") as f:
        f.write(contents)
    with open(datpath, "w") as f:
        f.write("%% %s\n" % config.exe[0])
        f.write("%% Speedup versus number of cores (%s image)\n" % config.input_size)
        f.write("% Img size, Number of cores, runtime (sec), 20th percentile runtime, 80th pctile runtime, speedup, 20th pctile speedup, 80th pctile speedup:\n")
        singlerank = results["baseline"].timing.runtime
        for k, v in results.items():
            speedup = calc_speedup(singlerank, k, v.timing)
            nranks = str(k)
            if nranks == "baseline" or nranks == "1":
                cores = 2 * config.cores_per_socket
            else:
                cores = k * config.cores_per_socket
            if nranks == "baseline":
                # We want the baseline data in the file for safe
                # keeping, but putting it as an actual row screws up
                # the ncores datatype when parsing. So comment it out.
                cores = "baseline"
                f.write("% ")
            f.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (config.input_size, str(cores), str(v.timing.runtime),
                                                   str(v.timing.percentile20), str(v.timing.percentile80), str(speedup.runtime),
                                                   str(speedup.percentile20), str(speedup.percentile80)))

def parse_config_argv():
    parser = OptionParser("Usage: %prog [options] exe [args]")
    parser.add_option("-o", "--output_dir", dest="output_dir",
                      help="Directory to place raw run results.")
    parser.add_option("-s", "--src", dest="srcfile",
                      help="Source file for executable")
    parser.add_option("-n", "--nodes", dest="nodes",
                      help="Comma separated number of nodes to run tests on.")
    parser.add_option("-b", "--baseline", dest="baseline",
                      help="Specify a single rank runtime instead of running it.")
    (options, args) = parser.parse_args()
    if (len(args) == 0 or len(options.nodes) == 0):
        parser.print_help()
        sys.exit(1)
    config = Config(options.output_dir, options.nodes,
                    options.srcfile, options.baseline, args)
    return config

def main():
    config = parse_config_argv()
    try:
        results = run(config)
    except KeyboardInterrupt:
        execute(["skill", "-u", "tyler"])
        sys.exit(1)
    report(config, results)

if __name__ == "__main__":
    main()
