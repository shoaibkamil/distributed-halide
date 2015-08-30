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
        self.tasks_per_node = [1]
        self.srcfile = srcfile
        self.baseline = float(baseline) if baseline else None
        self.exe = args
        self.input_size = "%sx%s" % (args[-2], args[-1])
        if baseline == None and 1 not in self.nodes:
            self.nodes = [1] + self.nodes
        if 1 not in self.tasks_per_node:
            self.tasks_per_node = [1] + self.tasks_per_node

class Result:
    def __init__(self, cmd, num_nodes, value):
        self.cmd = cmd
        self.num_nodes = num_nodes
        self.value = value

def make_run_cmd(config, nranks, num_nodes):
    os.environ["MV2_ENABLE_AFFINITY"] = "0"
    cmd = ["srun", "--exclude=lanka11", "--exclusive"]
    if nranks == num_nodes * 2:
        cmd.extend(["--cpu_bind=verbose,sockets"])
    elif nranks > num_nodes:
        cmd.extend(["--cpu_bind=verbose,cores"])
    else:
        cmd.extend(["--cpu_bind=verbose"])
    cmd.extend(["--ntasks=%d" % nranks])
    cmd.extend(["--nodes=%d" % num_nodes])
    cmd.extend(config.exe)
    cmd = map(str, cmd)
    return cmd

def execute(cmd):
    print "Executing %s..." % " ".join(cmd)
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
    exp = "Timing: <\d+> ranks <(\d*\.\d+)> seconds"
    m = re.search(exp, output)
    if m:
        return float(m.groups()[0])
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
    for num_nodes in config.nodes:
        for ntasks in config.tasks_per_node:
            nranks = num_nodes * ntasks
            cmd = make_run_cmd(config, nranks, num_nodes)
            result = get_time(execute(cmd))
            results[nranks] = Result(cmd, num_nodes, result)
    if config.baseline:
        results["baseline"] = Result("Baseline specified on command line", 1, config.baseline)
    else:
        assert "baseline" not in results.keys()
        results["baseline"] = results[1]
    return results

def calc_speedup(singlerank, numranks, runtime):
    try:
        return singlerank/runtime
    except ZeroDivisionError:
        return numranks

def calculate_highest_speedup(results):
    ranks = max(filter(lambda x: isinstance(x, int), results.keys()))
    singlerank = results["baseline"].value
    return (ranks, calc_speedup(singlerank, ranks, results[ranks].value))

def report(config, results):
    best = calculate_highest_speedup(results)
    print "Baseline runtime: %.3f sec" % results["baseline"].value
    print "%d rank runtime: %.3f" % (best[0], results[best[0]].value)
    print "%d rank speedup: %.3f (%.1f%% of linear)" % (best[0], best[1], (best[1]/best[0])*100.0)
    if config.output_dir == None:
        return
    contents = ""
    contents += config.exe[0] + "\n"
    contents += "Input size: %s\n" % config.input_size
    contents += "Testing node counts %s\n" % ", ".join(map(str, config.nodes))
    contents += "Invocation commands:\n"
    for k, v in results.items():
        contents += " %d: %s\n" % (k, " ".join(v.cmd))
    contents += "Runtime and speedup per rank count (in seconds):\n"
    contents += "--BEGIN DATA--\n"
    singlerank = results["baseline"].value
    for k, v in results.items():
        speedup = calc_speedup(singlerank, k, v.value)
        contents += "%d: %.3f\t%.3f\n" % (k, v.value, speedup)
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
        f.write("%% Speedup versus number of ranks (%s image)\n" % config.input_size)
        f.write("% Img size, Number of ranks, runtime (sec), speedup:\n")
        singlerank = results["baseline"].value
        for k, v in results.items():
            speedup = calc_speedup(singlerank, k, v.value)
            f.write("%s,%s,%s,%s\n" % (config.input_size, str(k), str(v.value), str(speedup)))

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
