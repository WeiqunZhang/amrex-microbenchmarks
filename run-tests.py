#!/usr/bin/env python

import argparse, os, subprocess, sys

if len(sys.argv) < 2:
    print("Device type not specified.  Run `./run-tests.py -h` for options")

parser = argparse.ArgumentParser(description="GPU tests")
parser.add_argument("--device", type=str, default="", choices=['cuda','hip','dpcpp'])
parser.add_argument("--single_precision", action="store_true")
parser.add_argument("--clean", action="store_true")
args = parser.parse_args(sys.argv[1:])

tests  = [{'name':'mb'     ,'dir':'daxpy'        ,'time':[]},
          {'name':'cb'     ,'dir':'compute-bound','time':[]},
          {'name':'br'     ,'dir':'branch'       ,'time':[]},
          {'name':'dbl sum','dir':'reduce'       ,'time':[]},
          {'name':'max'    ,'dir':'max'          ,'time':[]},
          {'name':'scn'    ,'dir':'scan'         ,'time':[]},
          {'name':'jac'    ,'dir':'jacobi'       ,'time':[]},
          {'name':'jsy'    ,'dir':'jacobi_sync'  ,'time':[]},
          {'name':'jex'    ,'dir':'jacobi_elixir','time':[]},
          {'name':'jmp'    ,'dir':'jacobi_memorypool','time':[]},
          {'name':'aos smp','dir':'aos_simple'   ,'time':[]},
          {'name':'aos sha','dir':'aos_shared'   ,'time':[]},
          {'name':'gsrb'   ,'dir':'gsrb'         ,'time':[]}]

max_grid_size = ['256','128','64','32']
ntrys = 3

TOP = os.getcwd()

if args.clean:
    for test in tests:
        os.chdir(os.path.join(TOP,test['dir']))
        command = 'make clean'
        p0 = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=True)
        stdout0, stderr0 = p0.communicate()
        p0.stdout.close()
        p0.stderr.close()
    sys.exit(0)

command = 'make '
if (args.device == 'cuda'):
    command += 'USE_CUDA=TRUE'
elif (args.device == 'hip'):
    command += 'USE_HIP=TRUE'
elif (args.device == 'dpcpp'):
    command += 'USE_DPCPP=TRUE'
else:
    print("Device not supported")
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
if args.device == 'hip':
    executable = 'AMD_DIRECT_DISPATCH=1 ' + executable

for test in tests:
    os.chdir(os.path.join(TOP,test['dir']))
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
        test['time'].append(t)

for test in tests:
    print(f" |{test['name']:>9}", end='')
print(' |')
for imgs in range(len(max_grid_size)):
    for test in tests:
        print(f" |{test['time'][imgs]:>9.2e}", end='')
    print(' |')
