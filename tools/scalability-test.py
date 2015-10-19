#!/usr/bin/env python

import math
import numpy as np
from optparse import OptionParser
import os
import re
import subprocess
import sys
import time

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

hopper_torque_template = """
#PBS -q regular
#PBS -l mppwidth=%(mppwidth)d
#PBS -l walltime=%(walltime)s
#PBS -o %(outputfile)s
#PBS -j oe

cd $PBS_O_WORKDIR

module swap PrgEnv-pgi PrgEnv-gnu
module unload atp
module unload cray-shem
module load zlib
module load gcc

# Set environment variables after module loading.

export HL_JIT_TARGET=host
export HL_DISABLE_DISTRIBUTED=%(disabledistributed)s
export LD_LIBRARY_PATH=${PBS_O_WORKDIR}:${LD_LIBRARY_PATH}
export XTPE_LINK_TYPE=dynamic
export MPICH_MAX_THREAD_SAFETY=multiple

aprun -n %(mpitasks)d -N %(taskspernode)d -d %(threadspertask)d ./%(exefile)s
"""

def get_unique_path(template):
    i = 0
    while True:
        path = template % i
        i += 1
        if not os.path.exists(path):
            return path
    sys.exit(1)

def make_run_cmd_lanka(config, num_nodes, baseline=False):
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
    return cmd, ""

def predict_minutes(input_size, num_nodes):
    width, height = map(int, input_size.split("x"))
    # Say one iteration takes 0.8 seconds at width 23000.
    iter_sec = (0.8*width*(height/float(num_nodes/2.0)))/(23000.0*23000.0)
    # Say we do 100 iterations
    sec = 100 * iter_sec
    # Round up minutes, with a minimum of one minute
    minutes = int(math.ceil(max(sec, 60) / 60.0))
    # Add 5 minutes for initialization time
    minutes += 5
    print "predict %d seconds" % int(minutes*60)
    return minutes

def format_walltime(minutes):
    minimum = 3
    minutes = max(minutes, minimum)
    h, m = minutes / 60, minutes % 60
    return "%d:%d:00" % (h, m)

def make_run_cmd_hopper(config, num_nodes, baseline=False):
    appname = os.path.basename(config.exe[0])
    mppwidth = num_nodes * 24
    outputfile = get_unique_path(appname + "_" + config.input_size + ".%d.out")
    disabledistributed = 1 if baseline else 0
    mpitasks = 1 if baseline else num_nodes * 2
    taskspernode = 1 if baseline else 2
    threadspertask = 24 if baseline else 12
    exefile = ' '.join(config.exe)
    minutes = predict_minutes(config.input_size, num_nodes)
    walltime = format_walltime(minutes)
    pbs_contents = hopper_torque_template % {"mppwidth": mppwidth, "outputfile": outputfile,
                                             "walltime": walltime,
                                             "disabledistributed": disabledistributed,
                                             "mpitasks": mpitasks, "taskspernode": taskspernode,
                                             "threadspertask": threadspertask, "exefile": exefile}
    pbsfile = get_unique_path(appname + "_" + config.input_size + ".%d.pbs")
    with open(pbsfile, "w") as f:
        f.write(pbs_contents)
    cmd = ["qsub", pbsfile]
    return cmd, outputfile

def make_run_cmd(config, num_nodes, baseline=False):
    # Return a tuple of (cmd, outputfile) to execute the given
    # configure. If 'outputfile' is an empty string, the time will be
    # parsed from the stdout of cmd. Otherwise, the run time will be
    # parsed from the given file.
    return make_run_cmd_hopper(config, num_nodes, baseline)

def execute(cmd, outputfile):
    print "Executing %s..." % (" ".join(cmd))
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if len(outputfile) == 0:
            return output
        else:
            while not os.path.exists(outputfile):
                time.sleep(7)
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
        return Timing(-1, -1, -1)

def run(config):
    results = {}
    if not config.baseline:
        # Run baseline of a single node
        cmd, outputfile = make_run_cmd(config, 1, baseline=True)
        result = get_time(execute(cmd, outputfile))
        print "Result was: %f sec" % result.runtime
        results["baseline"] = Result(["HL_DISABLE_DISTRIBUTED=1"] + cmd, 1, result)
    else:
        results["baseline"] = Result("Baseline specified on command line", 1,
                                     Timing(config.baseline,config.baseline,config.baseline))
    # Run all other tests.
    for num_nodes in config.nodes:
        nranks = num_nodes * config.tasks_per_node if num_nodes > 1 else 1
        cmd, outputfile = make_run_cmd(config, num_nodes)
        result = get_time(execute(cmd, outputfile))
        #result = Timing(-1, -1, -1)
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
    if config.srcfile:
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
        #execute(["skill", "-u", "tyler"])
        sys.exit(1)
    report(config, results)

if __name__ == "__main__":
    main()
