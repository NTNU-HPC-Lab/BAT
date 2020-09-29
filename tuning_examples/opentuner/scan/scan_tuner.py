#!/usr/bin/env python

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import EnumParameter
from opentuner.search.manipulator import BooleanParameter
from opentuner import MeasurementInterface
from opentuner import Result
from numba import cuda
import math
import json

start_path = '../../../src/programs'

class ScanTuner(MeasurementInterface):
    all_results = []

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """

        block_sizes = [2**i for i in range(4, 10)]
        grid_sizes = [2**i for i in range(0, 10)]
        manipulator = ConfigurationManipulator()

        manipulator.add_parameter(EnumParameter('BLOCK_SIZE', block_sizes))
        manipulator.add_parameter(EnumParameter('GRID_SIZE', grid_sizes)) 
        manipulator.add_parameter(EnumParameter('PRECISION', [32, 64]))
        manipulator.add_parameter(EnumParameter('UNROLL_LOOP_1', [0, 1]))
        manipulator.add_parameter(EnumParameter('UNROLL_LOOP_2', [0, 1]))
        manipulator.add_parameter(EnumParameter('USE_FAST_MATH', [0, 1]))
        manipulator.add_parameter(EnumParameter('OPTIMIZATION_LEVEL_HOST', ['-O0', '-O1', '-O2', '-O3']))
        manipulator.add_parameter(EnumParameter('OPTIMIZATION_LEVEL_DEVICE', ['-O0', '-O1', '-O2', '-O3']))
        manipulator.add_parameter(EnumParameter('MAX_REGISTERS', [-1, 20, 40, 60, 80, 100, 120]))
        manipulator.add_parameter(IntegerParameter('GPUS', 1, len(cuda.gpus)))

        return manipulator

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        args = argparser.parse_args()

        cfg = desired_result.configuration.data

        # Check constraints for the parameters
        if cfg['GRID_SIZE'] > cfg['BLOCK_SIZE']:
            return Result(time=float("inf"), state="ERROR", accuracy=float("-inf"))

        compute_capability = cuda.get_current_device().compute_capability
        cc = str(compute_capability[0]) + str(compute_capability[1])
        use_fast_math = ''
        if cfg['USE_FAST_MATH']:
            use_fast_math = '-use_fast_math '

        max_registers = f'-maxrregcount={cfg["MAX_REGISTERS"]} '
        if cfg['MAX_REGISTERS'] == -1:
            max_registers = ''

        make_program = f'nvcc {cfg["OPTIMIZATION_LEVEL_HOST"]} {use_fast_math}{max_registers}-Xptxas {cfg["OPTIMIZATION_LEVEL_DEVICE"]},-v -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -c {start_path}/scan/scan.cu'
        make_program += ' -D{0}={1}'.format('PRECISION',cfg['PRECISION'])
        make_program += ' -D{0}={1}'.format('UNROLL_LOOP_1',cfg['UNROLL_LOOP_1'])
        make_program += ' -D{0}={1}'.format('UNROLL_LOOP_2',cfg['UNROLL_LOOP_2'])
        make_program += ' -D{0}={1}'.format('GRID_SIZE',cfg['GRID_SIZE'])
        make_program += ' -D{0}={1} \n'.format('BLOCK_SIZE',cfg['BLOCK_SIZE'])

        if args.parallel == 1:
            # If parallel
            make_paralell_start = f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ {cfg["OPTIMIZATION_LEVEL_HOST"]} -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_paralell_end = f'mpicxx -L {start_path}/cuda-common -L {start_path}/common -o scan main.o scan.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
            compile_cmd = make_paralell_start + make_program + make_paralell_end
        elif args.parallel == 2:
            # If true parallel
            compile_cmd = f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ {cfg["OPTIMIZATION_LEVEL_HOST"]} -c -o main.o {start_path}/cuda-common/main.cpp \n'
            compile_cmd += f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ {cfg["OPTIMIZATION_LEVEL_HOST"]} -c -o tpScan.o {start_path}/scan/tpScan.cpp'
            compile_cmd += ' -D{0}={1}'.format('PRECISION',cfg['PRECISION'])
            compile_cmd += ' -D{0}={1}'.format('UNROLL_LOOP_1',cfg['UNROLL_LOOP_1'])
            compile_cmd += ' -D{0}={1}'.format('UNROLL_LOOP_2',cfg['UNROLL_LOOP_2'])
            compile_cmd += ' -D{0}={1}'.format('GRID_SIZE',cfg['GRID_SIZE'])
            compile_cmd += ' -D{0}={1} \n'.format('BLOCK_SIZE',cfg['BLOCK_SIZE'])
            compile_cmd += f'nvcc {cfg["OPTIMIZATION_LEVEL_HOST"]} {use_fast_math}{max_registers}-Xptxas {cfg["OPTIMIZATION_LEVEL_DEVICE"]},-v -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -c {start_path}/scan/tpScanLaunchKernel.cu'
            compile_cmd += ' -D{0}={1}'.format('PRECISION',cfg['PRECISION'])
            compile_cmd += ' -D{0}={1}'.format('UNROLL_LOOP_1',cfg['UNROLL_LOOP_1'])
            compile_cmd += ' -D{0}={1}'.format('UNROLL_LOOP_2',cfg['UNROLL_LOOP_2'])
            compile_cmd += ' -D{0}={1}'.format('GRID_SIZE',cfg['GRID_SIZE'])
            compile_cmd += ' -D{0}={1} \n'.format('BLOCK_SIZE',cfg['BLOCK_SIZE'])
            compile_cmd += f'mpicxx -L {start_path}/cuda-common -L {start_path}/common -o scan main.o tpScan.o tpScanLaunchKernel.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
        else:
            # If serial
            make_serial_start = f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ {cfg["OPTIMIZATION_LEVEL_HOST"]} -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_serial_end = f'nvcc -L {start_path}/cuda-common -L {start_path}/common -o scan main.o scan.o -lSHOCCommon'
            compile_cmd = make_serial_start + make_program + make_serial_end

        compile_result = self.call_program(compile_cmd)
        assert compile_result['returncode'] == 0

        program_command = './scan -s ' + str(args.size)
        if args.parallel == 1 or args.parallel == 2:
            chosen_gpu_number = cfg['GPUS']

            devices = ','.join([str(i) for i in range(0, chosen_gpu_number)])
            run_cmd = f'mpirun -np {chosen_gpu_number} --allow-run-as-root {program_command} -d {devices}'
        else: 
            run_cmd = program_command

        run_result = self.call_program(run_cmd)

        # Check that error code and error output is ok
        assert run_result['stderr'] == b''
        assert run_result['returncode'] == 0

        result = {'parameters': cfg, 'time': run_result['time']}
        self.all_results.append(result)
        return Result(time=run_result['time'])

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal parameter values written to results.json:", configuration.data)
        with open('all-results.json', 'w') as f:
            json.dump(self.all_results, f)
        self.manipulator().save_to_file(configuration.data, 'results.json')


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    argparser.add_argument('--size', type=int, default=1, help='problem size of the program (1-4)')
    argparser.add_argument('--gpu-num', type=int, default=1, help='number of GPUs')
    argparser.add_argument('--parallel', type=int, default=0, help='run on multiple GPUs (0=serial, 1=parallel, 2=true parallel)')
    ScanTuner.main(argparser.parse_args())
