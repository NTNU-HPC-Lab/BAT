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

class GccFlagsTuner(MeasurementInterface):

  def manipulator(self):
    """
    Define the search space by creating a
    ConfigurationManipulator
    """

    gpu = cuda.get_current_device()
    min_size = 1
    max_size = gpu.MAX_THREADS_PER_BLOCK

    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(EnumParameter('PRECISION', [32, 64]))
    # 0: ellpackr, 1: csr-normal-scalar, 2: csr-normal-vector, 3: csr-padded-scalar, 4: csr-padded-vector
    manipulator.add_parameter(EnumParameter('FORMAT', [0, 1, 2, 3, 4]))
    manipulator.add_parameter(IntegerParameter('BLOCK_SIZE', min_size, max_size))
    manipulator.add_parameter(EnumParameter('UNROLL_LOOP', [0, 1]))
    manipulator.add_parameter(EnumParameter('UNROLL_LOOP_2', [0, 1]))

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

    make_bfs = 'nvcc -gencode=arch=compute_' + cc + ',code=sm_' + cc + ' -I ' + start_path + '/cuda-common -I ' + start_path + '/common -g -O2 -c ' + start_path + '/spmv/spmv.cu'
    make_bfs += ' -D{0}={1}'.format('PRECISION',cfg['PRECISION'])
    make_bfs += ' -D{0}={1}'.format('FORMAT',cfg['FORMAT'])
    make_bfs += ' -D{0}={1} \n'.format('BLOCK_SIZE',cfg['BLOCK_SIZE'])
    make_bfs += 'nvcc -gencode=arch=compute_' + cc + ',code=sm_' + cc + ' -I ' + start_path + '/cuda-common -I ' + start_path + '/common -g -O2 -c ' + start_path + '/spmv/spmv_kernel.cu'
    make_bfs += ' -D{0}={1}'.format('UNROLL_LOOP',cfg['UNROLL_LOOP'])
    make_bfs += ' -D{0}={1}'.format('UNROLL_LOOP_2',cfg['UNROLL_LOOP_2'])
    make_bfs += ' -D{0}={1} \n'.format('BLOCK_SIZE',cfg['BLOCK_SIZE'])

    if args.parallel: 
      make_paralell_start = 'mpicxx -I ' + start_path + '/common/ -I ' + start_path + '/cuda-common/ -I /usr/local/cuda/include -DPARALLEL  -I ' + start_path + '/mpi-common/ -g -O2 -c -o main.o ' + start_path + '/cuda-common/main.cpp \n'
      make_paralell_end = 'mpicxx -L ' + start_path + '/cuda-common -L ' + start_path + '/common -o spmv main.o spmv.o spmv_kernel.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
      compile_cmd = make_paralell_start + make_bfs + make_paralell_end
    else: 
      make_serial_start = 'nvcc -I ' + start_path + '/common/ -I ' + start_path + '/cuda-common/ -g -O2 -c -o main.o ' + start_path + '/cuda-common/main.cpp \n'
      make_serial_end = 'nvcc -L ' + start_path + '/cuda-common -L ' + start_path + '/common -o spmv main.o spmv.o spmv_kernel.o -lSHOCCommon'
      compile_cmd = make_serial_start + make_bfs + make_serial_end
    
    compile_result = self.call_program(compile_cmd)
    assert compile_result['returncode'] == 0

    bfs_command = './spmv -s ' + str(args.problem_size)
    if args.parallel: 
      chosen_gpu_number = args.gpu_num
      if chosen_gpu_number > len(cuda.gpus):
        chosen_gpu_number = len(cuda.gpus)
      
      devices = ','.join([str(i) for i in range(0, chosen_gpu_number)])
      run_cmd = 'mpirun -np ' + str(chosen_gpu_number) + ' --allow-run-as-root ' + bfs_command + ' -d ' + devices
    else: 
      run_cmd = bfs_command

    run_result = self.call_program(run_cmd)
    assert run_result['returncode'] == 0

    return Result(time=run_result['time'])

  def save_final_config(self, configuration):
    """called at the end of tuning"""
    print("Optimal parameter values written to final_config.json:", configuration.data)
    self.manipulator().save_to_file(configuration.data, 'final_config.json')


if __name__ == '__main__':
  argparser = opentuner.default_argparser()
  argparser.add_argument('--problem-size', type=int, default=1, help='problem size of the program (1-4)')
  argparser.add_argument('--gpu-num', type=int, default=1, help='number of GPUs')
  argparser.add_argument('--parallel', type=bool, default=False, help='run on multiple GPUs')
  GccFlagsTuner.main(argparser.parse_args())
