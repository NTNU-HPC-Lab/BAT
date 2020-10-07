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

class SortTuner(MeasurementInterface):
    all_results = []

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """

        gpu = cuda.get_current_device()
        max_size = gpu.MAX_THREADS_PER_BLOCK
        # Using 2^i values less than `gpu.MAX_THREADS_PER_BLOCK` and over 16
        block_sizes = list(filter(lambda x: x <= max_size, [2**i for i in range(4, 11)]))

        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(IntegerParameter('LOOP_UNROLL_LSB', 0, 1))
        manipulator.add_parameter(IntegerParameter('LOOP_UNROLL_LOCAL_MEMORY', 0, 1))
        manipulator.add_parameter(IntegerParameter('LOOP_UNROLL_ADD_UNIFORM', 0, 1))
        manipulator.add_parameter(EnumParameter('SCAN_DATA_SIZE', [2, 4, 8]))
        manipulator.add_parameter(EnumParameter('SORT_DATA_SIZE', [2, 4, 8]))
        manipulator.add_parameter(EnumParameter('SCAN_BLOCK_SIZE', block_sizes))
        manipulator.add_parameter(EnumParameter('SORT_BLOCK_SIZE', block_sizes))
        manipulator.add_parameter(IntegerParameter('INLINE_LSB', 0, 1))
        manipulator.add_parameter(IntegerParameter('INLINE_SCAN', 0, 1))
        manipulator.add_parameter(IntegerParameter('INLINE_LOCAL_MEMORY', 0, 1))

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

        # Check constraint for block sizes and data sizes
        if cfg['SCAN_BLOCK_SIZE'] / cfg['SORT_BLOCK_SIZE'] != cfg['SORT_DATA_SIZE'] / cfg['SCAN_DATA_SIZE']:
            return Result(time=float("inf"), state="ERROR", accuracy=float("-inf"))

        # Constraint to ensure not attempting to use too much shared memory
        # 4 is the size of uints and 2 is because shared memory is used for both keys and values in the "reorderData" function
        # 16 * 2 is also added due to two other shared memory uint arrays used for offsets
        shared_memory_needed = (cfg['SCAN_BLOCK_SIZE'] * cfg['SCAN_DATA_SIZE'] * 4 * 2) + (4 * 16 * 2)
        gpu = cuda.get_current_device()
        available_shared_memory = gpu.MAX_SHARED_MEMORY_PER_BLOCK

        if shared_memory_needed > available_shared_memory:
            return Result(time=float("inf"), state="ERROR", accuracy=float("-inf"))
        
        make_program = f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -g -O2 -c {start_path}/sort/sort.cu'
        make_program += ' -D{0}={1}'.format('SCAN_DATA_SIZE', cfg['SCAN_DATA_SIZE'])
        make_program += ' -D{0}={1}'.format('SORT_DATA_SIZE', cfg['SORT_DATA_SIZE'])
        make_program += ' -D{0}={1}'.format('SCAN_BLOCK_SIZE', cfg['SCAN_BLOCK_SIZE'])
        make_program += ' -D{0}={1} \n'.format('SORT_BLOCK_SIZE', cfg['SORT_BLOCK_SIZE'])
        make_program += f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -g -O2 -c {start_path}/sort/sort_kernel.cu'
        make_program += ' -D{0}={1}'.format('LOOP_UNROLL_LSB', cfg['LOOP_UNROLL_LSB'])
        make_program += ' -D{0}={1}'.format('LOOP_UNROLL_LOCAL_MEMORY', cfg['LOOP_UNROLL_LOCAL_MEMORY'])
        make_program += ' -D{0}={1}'.format('LOOP_UNROLL_ADD_UNIFORM', cfg['LOOP_UNROLL_ADD_UNIFORM'])
        make_program += ' -D{0}={1}'.format('SCAN_DATA_SIZE', cfg['SCAN_DATA_SIZE'])
        make_program += ' -D{0}={1}'.format('SORT_DATA_SIZE', cfg['SORT_DATA_SIZE'])
        make_program += ' -D{0}={1}'.format('SCAN_BLOCK_SIZE', cfg['SCAN_BLOCK_SIZE'])
        make_program += ' -D{0}={1}'.format('SORT_BLOCK_SIZE', cfg['SORT_BLOCK_SIZE'])
        make_program += ' -D{0}={1}'.format('INLINE_LSB', cfg['INLINE_LSB'])
        make_program += ' -D{0}={1}'.format('INLINE_SCAN', cfg['INLINE_SCAN'])
        make_program += ' -D{0}={1} \n'.format('INLINE_LOCAL_MEMORY', cfg['INLINE_LOCAL_MEMORY'])

        if args.parallel:
            make_paralell_start = f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -g -O2 -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_paralell_end = f'mpicxx -L {start_path}/cuda-common -L {start_path}/common -o sort main.o sort.o sort_kernel.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
            compile_cmd = make_paralell_start + make_program + make_paralell_end
        else:
            make_serial_start = f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ -g -O2 -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_serial_end = f'nvcc -L {start_path}/cuda-common -L {start_path}/common -o sort main.o sort.o sort_kernel.o -lSHOCCommon'
            compile_cmd = make_serial_start + make_program + make_serial_end
        
        compile_result = self.call_program(compile_cmd)
        assert compile_result['returncode'] == 0

        program_command = './sort -s ' + str(args.size)
        if args.parallel:
            # Select number below max connected GPUs
            chosen_gpu_number = min(args.gpu_num, len(cuda.gpus))
      
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
            json.dump(self.all_results, f, indent=4)

        # Update configuration with problem size and tuning technique
        configuration.data["PROBLEM_SIZE"] = argparser.parse_args().size
        configuration.data["TUNING_TECHNIQUE"] = argparser.parse_args().technique

        self.manipulator().save_to_file(configuration.data, 'results.json')


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    argparser.add_argument('--size', type=int, default=1, help='problem size of the program (1-4)')
    argparser.add_argument('--gpu-num', type=int, default=1, help='number of GPUs')
    argparser.add_argument('--parallel', action="store_true", help='run on multiple GPUs')
    SortTuner.main(argparser.parse_args())
