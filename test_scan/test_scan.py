#!/usr/bin/env python3

import argparse, os, random, subprocess, sys, time

if len(sys.argv) < 2:
    print("Device type not specified.  Run `./test_scan.py -h` for options")

parser = argparse.ArgumentParser(description="Scan test")
parser.add_argument("--time", type=float, default=1)
parser.add_argument("--report", type=float, default=1)
parser.add_argument("--max_size", type=int, default=200000000)
parser.add_argument("--randomize_size", action="store_true")
parser.add_argument("--command", type=str, default="./main3d.gnu.TPROF.CUDA.ex")
args = parser.parse_args(sys.argv[1:])

tstart = time.perf_counter()
tmax = tstart + args.time*60

ntests_total = 0
ntests_pass = 0
ntests_amrex_fail = 0
ntests_thrust_fail = 0

t0 = time.perf_counter()
while time.perf_counter() < tmax:
    if args.randomize_size:
        command = args.command+' n='+str(random.randrange(1,args.max_size))
    else:
        command = args.command+' n='+str(args.max_size)
    p0 = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout0, stderr0 = p0.communicate()
    p0.stdout.close()
    p0.stderr.close()
    ntests_total += 1;
    for line in stdout0.decode().split('\n'):
        if line.startswith("pass"):
            ntests_pass += 1
        elif line.startswith("amrex failed"):
            ntests_amrex_fail += 1
        elif line.startswith("thrust failed"):
            ntests_thrust_fail += 1
    t1 = time.perf_counter()
    if (t1-t0 > args.report*60):
        t0 = t1
        print("After", int((t1-tstart)/60.), "minutes", ntests_pass, "out of", ntests_total, "tests passed")
        if (ntests_amrex_fail > 0):
            print("    amrex failed", ntests_amrex_fail, "times")
        if (ntests_thrust_fail > 0):
            print("    thrust failed", ntests_thrust_fail, "times")

print("Final:", ntests_pass, "out of", ntests_total, "tests passed")
if (ntests_amrex_fail > 0):
    print("    amrex failed", ntests_amrex_fail, "times")
if (ntests_thrust_fail > 0):
    print("    thrust failed", ntests_thrust_fail, "times")
