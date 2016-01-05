#!/usr/bin/env python

import numpy as np
from optparse import OptionParser
import os
import re
import subprocess
import sys
import time

class Config:
    def __init__(self, output_dir, nodes, srcfile, baseline, timelimits, fakerun, args):
        self.output_dir = output_dir
        self.nodes = map(int, nodes.split(","))
        self.tasks_per_node = 2
        self.cores_per_socket = 16
        self.srcfile = srcfile
        self.baseline = float(baseline) if baseline else None
        self.timelimit = timelimits.split(",")
        self.fakerun = fakerun
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

cori_slurm_template = """#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N %(numnodes)d
#SBATCH -t %(walltime)s
#SBATCH -o %(outputfile)s

export MV2_ENABLE_AFFINITY=0
export MPICH_MAX_THREAD_SAFETY=multiple
export HL_JIT_TARGET=host
export HL_DISABLE_DISTRIBUTED=%(disabledistributed)s
export XTPE_LINK_TYPE=dynamic

module load PrgEnv-gnu
module unload atp
module unload cray-shmem
module load zlib
module load python
module load numpy

srun -n %(mpitasks)d -c %(threadspertask)d %(cpubind)s %(exefile)s
"""

def get_unique_path(template):
    i = 0
    while True:
        path = template % i
        i += 1
        if not os.path.exists(path):
            return path
    sys.exit(1)

def make_run_cmd_cori(config, num_nodes, timelimit, baseline=False):
    appname = os.path.basename(config.exe[0])
    outputfile = get_unique_path(appname + "_" + config.input_size + ".%d.out")
    disabledistributed = 1 if baseline else 0
    mpitasks = 1 if baseline else num_nodes * 2
    taskspernode = 1 if baseline else 2
    threadspertask = 32 if baseline else config.cores_per_socket
    cpubind = "" if baseline else "--cpu_bind=verbose,sockets"
    exefile = ' '.join(config.exe)
    pbs_contents = cori_slurm_template % {"outputfile": outputfile,
                                          "walltime": timelimit,
                                          "disabledistributed": disabledistributed,
                                          "numnodes": num_nodes, "cpubind": cpubind,
                                          "mpitasks": mpitasks, "taskspernode": taskspernode,
                                          "threadspertask": threadspertask, "exefile": exefile}
    pbsfile = get_unique_path(appname + "_" + config.input_size + ".%d.sbatch")
    with open(pbsfile, "w") as f:
        f.write(pbs_contents)
    cmd = ["sbatch", pbsfile]
    return cmd, outputfile

def make_run_cmd(config, num_nodes, timelimit, baseline=False):
    return make_run_cmd_cori(config, num_nodes, timelimit, baseline)

def parse_job_id(output):
    # Submitted batch job 813839
    exp = "Submitted batch job (\d+)"
    m = re.search(exp, output)
    if m:
        jobid = int(m.groups()[0])
        return jobid
    else:
        return -1

def job_is_done(jobid):
    cmd = "sacct -n -j %d.0 -o State" % jobid
    try:
        output = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT)
        if (re.search("COMPLETED", output) or
            re.search("FAILED", output) or
            re.search("CANCELLED", output)):
            return True
        else:
            return False
    except subprocess.CalledProcessError as e:
        print e
        return True

def execute(cmd, outputfile):
    print "Executing %s..." % (" ".join(cmd))
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        jobid = parse_job_id(output)
        if len(outputfile) == 0:
            return output
        else:
            while not job_is_done(jobid):
                time.sleep(6)
            with open(outputfile, "r") as f:
                output = f.read()
            return output
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
        print "No time match from "
        print output
        return -1

def run(config):
    results = {}
    if not config.baseline:
        # Run baseline of a single node
        cmd, outputfile = make_run_cmd(config, 1, config.timelimit[0], baseline=True)
        if not config.fakerun:
            result = get_time(execute(cmd, outputfile))
            print "Result was: %f sec" % result.runtime
            results["baseline"] = Result(["HL_DISABLE_DISTRIBUTED=1"] + cmd, 1, result)
    else:
        results["baseline"] = Result("Baseline specified on command line", 1, config.baseline)
    # Run all other tests.
    for i, num_nodes in enumerate(config.nodes):
        nranks = num_nodes * config.tasks_per_node if num_nodes > 1 else 1
        timelimit_idx = min(i, len(config.timelimit)-1)
        cmd, outputfile = make_run_cmd(config, num_nodes, config.timelimit[timelimit_idx])
        if not config.fakerun:
            result = get_time(execute(cmd, outputfile))
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
    if config.srcfile:
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
    parser.add_option("-t", "--timelimit", dest="timelimit",
                      default="00:10:00",
                      help="Specify a maximum wall clock time for the job.")
    parser.add_option("-f", "--fake-run", dest="fakerun",
                      action="store_true", default=False,
                      help="Just create batch files, don't actually submit them.")
    (options, args) = parser.parse_args()
    if (len(args) == 0 or len(options.nodes) == 0):
        parser.print_help()
        sys.exit(1)
    config = Config(options.output_dir, options.nodes,
                    options.srcfile, options.baseline,
                    options.timelimit, options.fakerun, args)
    return config

def main():
    config = parse_config_argv()
    results = run(config)
    if not config.fakerun:
        report(config, results)

if __name__ == "__main__":
    main()
