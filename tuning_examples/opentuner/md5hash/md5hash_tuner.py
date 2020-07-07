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

class MD5HashTuner(MeasurementInterface):

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        """
        gpu = cuda.get_current_device()
        sizes = [1000, 10000, 100000, 1000000, 10000000]
        numVerts = sizes[argparser.parse_args().problem_size - 1]

        min_size = 1
        max_size = min(numVerts, gpu.MAX_THREADS_PER_BLOCK)
        """
        manipulator = ConfigurationManipulator()
        #manipulator.add_parameter(IntegerParameter('BLOCK_SIZE', min_size, max_size))
        #manipulator.add_parameter(EnumParameter('CHUNK_SIZE', [32, 64, 128, 256]))
        manipulator.add_parameter(EnumParameter('BLOCK_SIZE', [252, 384]))
        manipulator.add_parameter(EnumParameter('ROUND_STYLE', [0, 1]))


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

        make_program = 'nvcc -gencode=arch=compute_{0},code=sm_{0} -I {1}/cuda-common -I {1}/common -g -O2 -c {1}/md5hash/md5hash.cu'.format(cc, start_path)
        make_program += ' -D{0}={1}'.format('ROUND_STYLE',cfg['ROUND_STYLE'])
        make_program += ' -D{0}={1} \n'.format('BLOCK_SIZE',cfg['BLOCK_SIZE'])

        if args.parallel:
            make_paralell_start = 'mpicxx -I {0}/common/ -I {0}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {0}/mpi-common/ -g -O2 -c -o main.o {0}/cuda-common/main.cpp \n'.format(start_path)
            make_paralell_end = 'mpicxx -L {0}/cuda-common -L {0}/common -o md5hash main.o md5hash.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'.format(start_path)
            compile_cmd = make_paralell_start + make_program + make_paralell_end
        else:
            make_serial_start = 'nvcc -I {0}/common/ -I {0}/cuda-common/ -g -O2 -c -o main.o {0}/cuda-common/main.cpp \n'.format(start_path)
            make_serial_end = 'nvcc -L {0}/cuda-common -L {0}/common -o md5hash main.o md5hash.o -lSHOCCommon'.format(start_path)
            compile_cmd = make_serial_start + make_program + make_serial_end
    
        compile_result = self.call_program(compile_cmd)
        assert compile_result['returncode'] == 0

        program_command = './md5hash -s ' + str(args.problem_size)
        if args.parallel:
            # Select number below max connected GPUs
            chosen_gpu_number = min(args.gpu_num, len(cuda.gpus))
      
            devices = ','.join([str(i) for i in range(0, chosen_gpu_number)])
            run_cmd = 'mpirun -np {0} --allow-run-as-root {1} -d {2}'.format(chosen_gpu_number, program_command, devices)
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
  argparser.add_argument('--parallel', type=bool, default=False, help='run on multiple GPUs')
  MD5HashTuner.main(argparser.parse_args())