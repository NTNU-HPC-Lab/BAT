#!/usr/bin/env python

from __future__ import print_function
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

class TriadTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """

        # TODO: python triad_tuner.py --no-dups

        gpu = cuda.get_current_device()        
        max_size = gpu.MAX_THREADS_PER_BLOCK
        # Using 2^i values less than `gpu.MAX_THREADS_PER_BLOCK`
        block_sizes = list(filter(lambda x: x <= max_size, [2**i for i in range(0, 11)]))

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

        make_command = 'nvcc -gencode=arch=compute_{0},code=sm_{0} -I {1}/cuda-common -I {1}/common -g -O2 -c {1}/triad/triad.cu'.format(cc, start_path)
        make_command += ' -D{0}={1} \n'.format('BLOCK_SIZE', cfg['BLOCK_SIZE'])
        make_command += 'nvcc -gencode=arch=compute_{0},code=sm_{0} -I {1}/cuda-common -I {1}/common -g -O2 -c {1}/triad/triad_kernel.cu\n'.format(cc, start_path)

        if args.parallel:
            make_paralell_start = 'mpicxx -I ' + start_path + '/common/ -I ' + start_path + '/cuda-common/ -I /usr/local/cuda/include -DPARALLEL  -I ' + start_path + '/mpi-common/ -g -O2 -c -o main.o ' + start_path + '/cuda-common/main.cpp \n'
            make_paralell_end = 'mpicxx -L ' + start_path + '/cuda-common -L ' + start_path + '/common -o triad main.o triad.o triad_kernel.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
            compile_cmd = make_paralell_start + make_command + make_paralell_end
        else:
            make_serial_start = 'nvcc -I ' + start_path + '/common/ -I ' + start_path + '/cuda-common/ -g -O2 -c -o main.o ' + start_path + '/cuda-common/main.cpp \n'
            make_serial_end = 'nvcc -L ' + start_path + '/cuda-common -L ' + start_path + '/common -o triad main.o triad.o triad_kernel.o -lSHOCCommon'
            compile_cmd = make_serial_start + make_command + make_serial_end
        
        compile_result = self.call_program(compile_cmd)
        assert compile_result['returncode'] == 0

        triad_command = './triad -s ' + str(args.problem_size)
        if args.parallel: 
            chosen_gpu_number = args.gpu_num
            if chosen_gpu_number > len(cuda.gpus):
                chosen_gpu_number = len(cuda.gpus)
      
            devices = ','.join([str(i) for i in range(0, chosen_gpu_number)])
            run_cmd = 'mpirun -np ' + str(chosen_gpu_number) + ' --allow-run-as-root ' + triad_command + ' -d ' + devices
        else:
                run_cmd = triad_command

        run_result = self.call_program(run_cmd)
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
    TriadTuner.main(argparser.parse_args())
