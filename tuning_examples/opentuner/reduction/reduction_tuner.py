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

class ReductionTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """

        gpu = cuda.get_current_device()        
        max_size = gpu.MAX_THREADS_PER_BLOCK
        # Using 2^i values less than `gpu.MAX_THREADS_PER_BLOCK`
        block_sizes = list(filter(lambda x: x <= max_size, [1, 2, 4, 8, 16, 64, 128, 256, 512]))
        # TODO: check why it does not work for block size = 32

        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(EnumParameter('BLOCK_SIZE', block_sizes))

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

        make_program = 'nvcc -gencode=arch=compute_{0},code=sm_{0} -I {1}/cuda-common -I {1}/common -g -O2 -c {1}/reduction/reduction.cu'.format(cc, start_path)
        make_program += ' -D{0}={1} \n'.format('BLOCK_SIZE', cfg['BLOCK_SIZE'])

        if args.parallel == 1:
            # If parallel
            make_paralell_start = 'mpicxx -I {0}/common/ -I {0}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {0}/mpi-common/ -g -O2 -c -o main.o {0}/cuda-common/main.cpp \n'.format(start_path)
            make_paralell_end = 'mpicxx -L {0}/cuda-common -L {0}/common -o -o reduction main.o reduction.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'.format(start_path)
            compile_cmd = make_paralell_start + make_program + make_paralell_end
        elif args.parallel == 2:
            # If true parallel
            compile_cmd = 'mpicxx -I {0}/common/ -I {0}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {0}/mpi-common/ -g -O2 -c -o main.o {0}/cuda-common/main.cpp \n'.format(start_path)
            compile_cmd += 'mpicxx -I {0}/common/ -I {0}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {0}/mpi-common/ -g -O2 -c -o tpReduction.o {0}/reduction/tpReduction.cpp'.format(start_path)
            compile_cmd += ' -D{0}={1} \n'.format('BLOCK_SIZE', cfg['BLOCK_SIZE'])
            compile_cmd += 'nvcc -gencode=arch=compute_{0},code=sm_{0} -I {1}/cuda-common -I {1}/common -g -O2 -c {1}/reduction/tpRedLaunchKernel.cu'.format(cc, start_path)
            compile_cmd += ' -D{0}={1} \n'.format('BLOCK_SIZE', cfg['BLOCK_SIZE'])
            compile_cmd += 'mpicxx -L {0}/cuda-common -L {0}/common -o reduction main.o tpReduction.o tpRedLaunchKernel.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'.format(start_path)
        else:
            # If serial
            make_serial_start = 'nvcc -I {0}/common/ -I {0}/cuda-common/ -g -O2 -c -o main.o {0}/cuda-common/main.cpp \n'.format(start_path)
            make_serial_end = 'nvcc -L {0}/cuda-common -L {0}/common -o reduction main.o reduction.o -lSHOCCommon'.format(start_path)
            compile_cmd = make_serial_start + make_program + make_serial_end

        compile_result = self.call_program(compile_cmd)
        assert compile_result['returncode'] == 0

        program_command = './reduction -s ' + str(args.problem_size)
        if args.parallel == 0:
            run_cmd = program_command
        else:
            # Select number below max connected GPUs
            chosen_gpu_number = min(args.gpu_num, len(cuda.gpus))
      
            devices = ','.join([str(i) for i in range(0, chosen_gpu_number)])
            run_cmd = 'mpirun -np {0} --allow-run-as-root {1} -d {2}'.format(chosen_gpu_number, program_command, devices)

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
    argparser.add_argument('--parallel', type=int, default=0, help='run on multiple GPUs (0=serial, 1=parallel, 2=true parallel)')
    ReductionTuner.main(argparser.parse_args())
