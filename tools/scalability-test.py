#!/usr/bin/python

from optparse import OptionParser
import os
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

class Result:
    def __init__(self, cmd, value):
        self.cmd = cmd
        self.value = value
            
def make_run_cmd(config, num_nodes, tasks_per_node):
    cmd = ["srun", "--exclude=lanka11"]
    cmd.extend(["--cpu_bind=verbose"])
    cmd.extend(["-n", tasks_per_node])
    cmd.extend(["-N", num_nodes])
    cmd.extend(config.exe)
    cmd = map(str, cmd)
    return cmd

def execute(cmd):
    return subprocess.check_output(cmd)

def readall(path):
    with open(path, "r") as f:
        contents = f.read()
    return contents

def get_time(output):
    exp = "Timing: <\d+> ranks <(\d*\.\d+)> seconds"
    m = re.search(exp, output)
    if m:
        return m.groups()[1]
    else:
        return -1

def run(config):
    results = {}
    for num_nodes in config.nodes:
        tasks_per_node = 1
        cmd = make_run_cmd(config, num_nodes, tasks_per_node)
        result = get_time(execute(cmd))
        results[num_nodes] = Result(cmd, result)
    return results

def report(config, results):
    print config.exe[0]
    print "Testing node counts %s" % config.nodes
    print "Invocation commands:"
    for k, v in results:
        print "    %d: %s" % (k, v.cmd)
    print "End to end times per rank count (in seconds):"
    print "--BEGIN DATA--"
    for k, v in results:
        print "%d: %.3f" % (k, v.value)
    print "--END DATA--"
    print "Raw source dump:"
    print "// File name %s" % config.srcfile
    print readall(config.srcfile)
    
def parse_config_argv():
    parser = OptionParser("Usage: %prog [options] exe [args]")
    parser.add_option("-o", "--output_dir", dest="output_dir",
                      help="Directory to place raw run results.")
    parser.add_option("-n", "--nodes", dest="nodes",
                      help="Comma separated number of nodes to run tests on.")
    parser.add_option("-s", "--src", dest="srcfile",
                      help="Source file for executable")
    (options, args) = parser.parse_args()
    config = Config(options.output_dir, options.nodes,
                    options.srcfile, args)
    return config

def main():
    config = parse_config_argv()
    results = run(config)
    report(config, results)
    
if __name__ == "__main__":
    main()
