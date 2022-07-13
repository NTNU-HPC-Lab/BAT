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

class MD5HashTuner(MeasurementInterface):
    all_results = []

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """

        gpu = cuda.get_current_device()
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(IntegerParameter('BLOCK_SIZE', 1, gpu.MAX_THREADS_PER_BLOCK))
        manipulator.add_parameter(IntegerParameter('ROUND_STYLE', 0, 1))
        manipulator.add_parameter(IntegerParameter('UNROLL_LOOP_1', 0, 1))
        manipulator.add_parameter(IntegerParameter('UNROLL_LOOP_2', 0, 1))
        manipulator.add_parameter(IntegerParameter('UNROLL_LOOP_3', 0, 1))
        manipulator.add_parameter(IntegerParameter('INLINE_1', 0, 1))
        manipulator.add_parameter(IntegerParameter('INLINE_2', 0, 1))
        manipulator.add_parameter(IntegerParameter('WORK_PER_THREAD_FACTOR', 1, 5))

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

        cfg = {
                "ROUND_STYLE": 0,
                "UNROLL_LOOP_1": 1,
                "UNROLL_LOOP_2": 1,
                "UNROLL_LOOP_3": 1,
                "INLINE_1": 0,
                "INLINE_2": 0,
                "WORK_PER_THREAD_FACTOR": 1,
                "BLOCK_SIZE": 256
        }

        make_program = f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -g -O2 -c {start_path}/md5hash/md5hash.cu'
        make_program += ' -D{0}={1}'.format('ROUND_STYLE',cfg['ROUND_STYLE'])
        make_program += ' -D{0}={1}'.format('UNROLL_LOOP_1',cfg['UNROLL_LOOP_1'])
        make_program += ' -D{0}={1}'.format('UNROLL_LOOP_2',cfg['UNROLL_LOOP_2'])
        make_program += ' -D{0}={1}'.format('UNROLL_LOOP_3',cfg['UNROLL_LOOP_3'])
        make_program += ' -D{0}={1}'.format('INLINE_1',cfg['INLINE_1'])
        make_program += ' -D{0}={1}'.format('INLINE_2',cfg['INLINE_2'])
        make_program += ' -D{0}={1}'.format('WORK_PER_THREAD_FACTOR',cfg['WORK_PER_THREAD_FACTOR'])
        make_program += ' -D{0}={1} \n'.format('BLOCK_SIZE',cfg['BLOCK_SIZE'])

        if args.parallel:
            make_paralell_start = f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -g -O2 -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_paralell_end = f'mpicxx -L {start_path}/cuda-common -L {start_path}/common -o md5hash main.o md5hash.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
            compile_cmd = make_paralell_start + make_program + make_paralell_end
        else:
            make_serial_start = f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ -g -O2 -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_serial_end = f'nvcc -L {start_path}/cuda-common -L {start_path}/common -o md5hash main.o md5hash.o -lSHOCCommon'
            compile_cmd = make_serial_start + make_program + make_serial_end
    
        compile_result = self.call_program(compile_cmd)
        assert compile_result['returncode'] == 0

        program_command = './md5hash -s ' + str(args.size)
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
  MD5HashTuner.main(argparser.parse_args())
