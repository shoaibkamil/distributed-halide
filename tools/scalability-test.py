#!/usr/bin/python

import numpy as np
from optparse import OptionParser
import os
import re
import subprocess
import sys

class Config:
    def __init__(self, output_dir, nodes, srcfile, args):
        self.output_dir = output_dir
        self.nodes = map(int, nodes.split(","))
        self.srcfile = srcfile
        self.exe = args
        if 1 not in self.nodes:
            self.nodes = [1] + self.nodes
        self.input_size = "%sx%s" % (args[-2], args[-1])

class Result:
    def __init__(self, cmd, value):
        self.cmd = cmd
        self.value = value

def make_run_cmd(config, num_nodes, tasks_per_node):
    cmd = ["srun", "--exclude=lanka11", "--exclusive"]
    cmd.extend(["--cpu_bind=verbose"])
    cmd.extend(["--ntasks-per-node", tasks_per_node])
    cmd.extend(["-N", num_nodes])
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
        tasks_per_node = 1
        cmd = make_run_cmd(config, num_nodes, tasks_per_node)
        result = get_time(execute(cmd))
        results[num_nodes] = Result(cmd, result)
    return results

def calculate_slope(results):
    x = []
    y = []
    singlenode = results[1].value
    for k, v in results.items():
        speedup = singlenode/v.value
        x.append(k)
        y.append(speedup)
    line = np.polyfit(x, y, 1)
    return line[0]

def report(config, results):
    slope = calculate_slope(results)
    print "Best fit speedup slope: %.3f" % slope
    contents = ""
    contents += config.exe[0] + "\n"
    contents += "Input size: %s\n" % config.input_size
    contents += "Testing node counts %s\n" % ", ".join(map(str, config.nodes))
    contents += "Invocation commands:\n"
    for k, v in results.items():
        contents += " %d: %s\n" % (k, " ".join(v.cmd))
    contents += "Best fit speedup slope: %.3f\n" % slope
    contents += "End to end times per rank count (in seconds):\n"
    contents += "--BEGIN DATA--\n"
    for k, v in results.items():
        contents += "%d: %.3f\n" % (k, v.value)
    contents += "--END DATA--\n"
    contents += "Raw source dump:\n"
    contents += "// File name %s\n" % config.srcfile
    contents += readall(config.srcfile)

    exe = os.path.basename(config.exe[0])
    template = "%s-%s-%s-%%d.dat" % (exe, ",".join(map(str, config.nodes)),
                                     config.input_size)
    path = get_unique_path(os.path.join(config.output_dir, template))
    with open(path, "w") as f:
        f.write(contents)

def parse_config_argv():
    parser = OptionParser("Usage: %prog [options] exe [args]")
    parser.add_option("-o", "--output_dir", dest="output_dir",
                      help="Directory to place raw run results.")
    parser.add_option("-n", "--nodes", dest="nodes",
                      help="Comma separated number of nodes to run tests on.")
    parser.add_option("-s", "--src", dest="srcfile",
                      help="Source file for executable")
    (options, args) = parser.parse_args()
    if (len(args) == 0 or len(options.output_dir) == 0 or len(options.nodes) == 0):
        parser.print_help()
        sys.exit(1)
    config = Config(options.output_dir, options.nodes,
                    options.srcfile, args)
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
