#!/usr/bin/env python3

#!/usr/bin/env python

import argparse, os, subprocess, sys, time

if len(sys.argv) < 2:
    print("Device type not specified.  Run `./test_scan.py -h` for options")

parser = argparse.ArgumentParser(description="Scan test")
parser.add_argument("--time", type=float)
parser.add_argument("--report", type=float, default=1)
parser.add_argument("--command", type=str, default="./main3d.gnu.TPROF.CUDA.ex")
args = parser.parse_args(sys.argv[1:])

tstart = time.perf_counter()
tmax = tstart + args.time*60

ntests_total = 0
ntests_pass = 0

t0 = time.perf_counter()
while time.perf_counter() < tmax:
    p0 = subprocess.Popen(args.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout0, stderr0 = p0.communicate()
    p0.stdout.close()
    p0.stderr.close()
    ntests_total += 1;
    for line in stdout0.decode().split('\n'):
        if line.startswith("pass"):
            ntests_pass += 1
    t1 = time.perf_counter()
    if (t1-t0 > args.report*60):
        t0 = t1
        print("After", int((t1-tstart)/60.), "minutes", ntests_pass, "out of", ntests_total, "tests passed")

print("Final:", ntests_pass, "out of", ntests_total, "tests passed")
