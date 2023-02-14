#!/usr/bin/env python3

import argparse, os, subprocess, sys

if len(sys.argv) < 2:
    print("Device type not specified.  Run `./run-single-test.py -h` for options")

parser = argparse.ArgumentParser(description="GPU tests")
parser.add_argument("--device", type=str, default="", choices=['cuda','hip','sycl','cpu'])
parser.add_argument("--test", type=str, default="")
parser.add_argument("--single_precision", action="store_true")
args = parser.parse_args(sys.argv[1:])

max_grid_size = ['256','128','64','32']
ntrys = 3

TOP = os.getcwd()

command = 'make '
if (args.device == 'cuda'):
    command += 'USE_CUDA=TRUE'
elif (args.device == 'hip'):
    command += 'USE_HIP=TRUE'
elif (args.device == 'sycl'):
    command += 'USE_SYCL=TRUE'
elif (args.device == 'cpu'):
    command += 'USE_CUDA=FALSE USE_HIP=FALSE USE_SYCL=FALSE'
else:
    print("Device not supported.  Device is either cuda or hip or sycl or cpu.")
    sys.exit(1)
if args.single_precision:
    command += ' PRECISION=FLOAT'
command += ' print-machineSuffix'

p0 = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
stdout0, stderr0 = p0.communicate()
p0.stdout.close()
p0.stderr.close()
for line in stdout0.decode().split('\n'):
    if line.startswith("machineSuffix is"):
        machineSuffix = line.split()[2]

executable = './main.'+machineSuffix+'.ex'

os.chdir(os.path.join(TOP,args.test))
ts = []
for mgs in max_grid_size:
    command = executable + ' max_grid_size=' + mgs
    t = 1.e100
    for itry in range(ntrys):
        print(os.getcwd(),": ", command)
        p0 = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=True)
        stdout0, stderr0 = p0.communicate()
        p0.stdout.close()
        p0.stderr.close()
        for line in stdout0.decode().split('\n'):
            if line.startswith("Kernel run time is"):
                t = min(t,float(line.split()[4][0:-1]))
    ts.append(t)

print(args.test+': ', end='')
for t in ts:
    print(f" {t:>9.2e}", end='')
print()
