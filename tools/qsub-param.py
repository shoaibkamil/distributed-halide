#!/usr/bin/env python

import math
import numpy as np
from optparse import OptionParser
import os
import re
import subprocess
import sys
import time

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
export HL_DISABLE_DISTRIBUTED=0
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

def replace_suffix(s, suffix, repl):
    if s.endswith(suffix):
        return s.replace(suffix, repl)
    return s

def make_run_cmd_hopper(config, num_nodes):
    appname = os.path.basename(config["exe"][0])
    mppwidth = num_nodes * 24
    pbsfile = get_unique_path(appname + "_quicksubmit.%d.pbs")
    outputfile = replace_suffix(pbsfile, ".pbs", ".out")
    mpitasks = num_nodes * 2
    taskspernode = 2
    threadspertask = 12
    exefile = ' '.join(config["exe"])
    walltime = config["timeout"]
    pbs_contents = hopper_torque_template % {"mppwidth": mppwidth, "outputfile": outputfile,
                                             "walltime": walltime,
                                             "mpitasks": mpitasks, "taskspernode": taskspernode,
                                             "threadspertask": threadspertask, "exefile": exefile}
    with open(pbsfile, "w") as f:
        f.write(pbs_contents)
    cmd = ["qsub", pbsfile]
    return cmd, outputfile

def execute(cmd, outputfile):
    print "Executing %s..." % (" ".join(cmd))
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print e
        return ""

def parse_config_argv():
    parser = OptionParser("Usage: %prog [options] exe [args]")
    parser.add_option("-n", "--nodes", dest="nodes",
                      help="Number of Hopper nodes to run on.")
    parser.add_option("-t", "--timeout", dest="timeout",
                      help="hh:mm:ss wall clock timeout. Default 00:10:00.",
                      default="00:10:00")

    (options, args) = parser.parse_args()
    if (len(args) == 0 or len(options.nodes) == 0):
        parser.print_help()
        sys.exit(1)
    config = {"nodes": int(options.nodes),
              "timeout": options.timeout,
              "exe": args}
    return config

def main():
    config = parse_config_argv()
    cmd, outputfile = make_run_cmd_hopper(config, config["nodes"])
    execute(cmd, outputfile)

if __name__ == "__main__":
    main()
