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

start_path = '../../../src/programs'

class SortTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """

        gpu = cuda.get_current_device()        
        max_size = gpu.MAX_THREADS_PER_BLOCK
        # Using 2^i values less than `gpu.MAX_THREADS_PER_BLOCK`
        block_sizes = list(filter(lambda x: x <= max_size, [2**i for i in range(0, 11)]))
        # TODO: use block size later

        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(EnumParameter('LOOP_UNROLL_LSB', [0, 1]))
        manipulator.add_parameter(EnumParameter('LOOP_UNROLL_LOCAL_MEMORY', [0, 1]))
        manipulator.add_parameter(EnumParameter('LOOP_UNROLL_ADD_UNIFORM', [0, 1]))

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

        make_program = f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -g -O2 -c {start_path}/sort/sort.cu\n'
        make_program += f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -g -O2 -c {start_path}/sort/sort_kernel.cu'
        make_program += ' -D{0}={1}'.format('LOOP_UNROLL_LSB', cfg['LOOP_UNROLL_LSB'])
        make_program += ' -D{0}={1}'.format('LOOP_UNROLL_LOCAL_MEMORY', cfg['LOOP_UNROLL_LOCAL_MEMORY'])
        make_program += ' -D{0}={1} \n'.format('LOOP_UNROLL_ADD_UNIFORM', cfg['LOOP_UNROLL_ADD_UNIFORM'])

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

        program_command = './sort -s ' + str(args.problem_size)
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

        return Result(time=run_result['time'])

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal parameter values written to results.json:", configuration.data)
        self.manipulator().save_to_file(configuration.data, 'results.json')


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    argparser.add_argument('--problem-size', type=int, default=1, help='problem size of the program (1-4)')
    argparser.add_argument('--gpu-num', type=int, default=1, help='number of GPUs')
    argparser.add_argument('--parallel', action="store_true", help='run on multiple GPUs')
    SortTuner.main(argparser.parse_args())