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

class ReductionTuner(MeasurementInterface):
    all_results = []
    
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        args = argparser.parse_args()
        max_gpus = len(cuda.gpus)
        if args.parallel == 0:
            max_gpus = 1

        gpu = cuda.get_current_device()
        max_block_size = gpu.MAX_THREADS_PER_BLOCK
        # Using 2^i values less than `gpu.MAX_THREADS_PER_BLOCK` except 32
        block_sizes = list(filter(lambda x: x <= max_block_size, [1, 2, 4, 8, 16, 64, 128, 256, 512, 1024]))

        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(EnumParameter('BLOCK_SIZE', block_sizes))
        manipulator.add_parameter(EnumParameter('GRID_SIZE', [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]))
        manipulator.add_parameter(EnumParameter('PRECISION', [32, 64]))
        manipulator.add_parameter(IntegerParameter('COMPILER_OPTIMIZATION_HOST', 0, 3))
        manipulator.add_parameter(IntegerParameter('COMPILER_OPTIMIZATION_DEVICE', 0, 3))
        manipulator.add_parameter(IntegerParameter('USE_FAST_MATH', 0, 1))
        manipulator.add_parameter(EnumParameter('MAX_REGISTERS', [-1, 20, 40, 60, 80, 100, 120]))
        manipulator.add_parameter(IntegerParameter('GPUS', 1, max_gpus))
        manipulator.add_parameter(IntegerParameter('LOOP_UNROLL_REDUCE_1', 0, 1))
        manipulator.add_parameter(IntegerParameter('LOOP_UNROLL_REDUCE_2', 0, 1))
        manipulator.add_parameter(IntegerParameter('TEXTURE_MEMORY', 0, 1))

        return manipulator

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        args = argparser.parse_args()

        cfg = desired_result.configuration.data
        compute_capability = cuda.get_current_device().compute_capability
        cc = str(compute_capability[0]) + str(compute_capability[1])

        use_fast_math = ''
        if cfg['USE_FAST_MATH'] == 1:
            use_fast_math = '-use_fast_math '

        max_registers = ''
        if cfg['MAX_REGISTERS'] != -1:
            max_registers = f'-maxrregcount={cfg["MAX_REGISTERS"]} '

        make_program = f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common {use_fast_math}{max_registers} -O{cfg["COMPILER_OPTIMIZATION_HOST"]} -Xptxas -O{cfg["COMPILER_OPTIMIZATION_DEVICE"]} -c {start_path}/reduction/reduction.cu'
        make_program += ' -D{0}={1}'.format('BLOCK_SIZE', cfg['BLOCK_SIZE'])
        make_program += ' -D{0}={1}'.format('GRID_SIZE', cfg['GRID_SIZE'])
        make_program += ' -D{0}={1}'.format('PRECISION', cfg['PRECISION'])
        make_program += ' -D{0}={1}'.format('LOOP_UNROLL_REDUCE_1', cfg['LOOP_UNROLL_REDUCE_1'])
        make_program += ' -D{0}={1}'.format('LOOP_UNROLL_REDUCE_2', cfg['LOOP_UNROLL_REDUCE_2'])
        make_program += ' -D{0}={1} \n'.format('TEXTURE_MEMORY', cfg['TEXTURE_MEMORY'])

        if args.parallel == 1:
            # If parallel
            make_paralell_start = f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -O{cfg["COMPILER_OPTIMIZATION_HOST"]} -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_paralell_end = f'mpicxx -L {start_path}/cuda-common -L {start_path}/common -o reduction main.o reduction.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
            compile_cmd = make_paralell_start + make_program + make_paralell_end
        elif args.parallel == 2:
            # If true parallel
            compile_cmd = f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -O{cfg["COMPILER_OPTIMIZATION_HOST"]} -c -o main.o {start_path}/cuda-common/main.cpp \n'
            compile_cmd += f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -O{cfg["COMPILER_OPTIMIZATION_HOST"]} -c -o tpReduction.o {start_path}/reduction/tpReduction.cpp'
            compile_cmd += ' -D{0}={1}'.format('BLOCK_SIZE', cfg['BLOCK_SIZE'])
            compile_cmd += ' -D{0}={1}'.format('GRID_SIZE', cfg['GRID_SIZE'])
            compile_cmd += ' -D{0}={1}'.format('PRECISION', cfg['PRECISION'])
            compile_cmd += ' -D{0}={1}'.format('LOOP_UNROLL_REDUCE_1', cfg['LOOP_UNROLL_REDUCE_1'])
            compile_cmd += ' -D{0}={1}'.format('LOOP_UNROLL_REDUCE_2', cfg['LOOP_UNROLL_REDUCE_2'])
            compile_cmd += ' -D{0}={1} \n'.format('TEXTURE_MEMORY', cfg['TEXTURE_MEMORY'])
            compile_cmd += f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common {use_fast_math}{max_registers} -O{cfg["COMPILER_OPTIMIZATION_HOST"]} -Xptxas -O{cfg["COMPILER_OPTIMIZATION_DEVICE"]} -c {start_path}/reduction/tpRedLaunchKernel.cu'
            compile_cmd += ' -D{0}={1} \n'.format('BLOCK_SIZE', cfg['BLOCK_SIZE'])
            compile_cmd += f'mpicxx -L {start_path}/cuda-common -L {start_path}/common -o reduction main.o tpReduction.o tpRedLaunchKernel.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
        else:
            # If serial
            make_serial_start = f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ -O{cfg["COMPILER_OPTIMIZATION_HOST"]} -Xptxas -O{cfg["COMPILER_OPTIMIZATION_DEVICE"]} -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_serial_end = f'nvcc -L {start_path}/cuda-common -L {start_path}/common -o reduction main.o reduction.o -lSHOCCommon'
            compile_cmd = make_serial_start + make_program + make_serial_end

        compile_result = self.call_program(compile_cmd)
        assert compile_result['returncode'] == 0

        program_command = './reduction -s ' + str(args.size)
        if args.parallel == 0:
            run_cmd = program_command
        else:
            # Select number in the range of 1 to max connected GPUs
            chosen_gpu_number = max(min(args.gpu_num if args.gpu_num is not None else cfg['GPUS'], len(cuda.gpus)), 1)

            devices = ','.join([str(i) for i in range(0, chosen_gpu_number)])
            run_cmd = f'mpirun -np {chosen_gpu_number} --allow-run-as-root {program_command} -d {devices}'

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
            json.dump(self.all_results, f, indent=4)
    
        # Update configuration with problem size and tuning technique
        configuration.data["PROBLEM_SIZE"] = argparser.parse_args().size
        configuration.data["TUNING_TECHNIQUE"] = argparser.parse_args().technique
        
        self.manipulator().save_to_file(configuration.data, 'results.json')


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    argparser.add_argument('--size', type=int, default=1, help='problem size of the program (1-4)')
    argparser.add_argument('--gpu-num', type=int, help='number of GPUs')
    argparser.add_argument('--parallel', type=int, default=0, help='run on multiple GPUs (0=serial, 1=parallel, 2=true parallel)')
    ReductionTuner.main(argparser.parse_args())
