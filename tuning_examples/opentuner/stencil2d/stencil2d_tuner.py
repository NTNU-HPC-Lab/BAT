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

class Stencil2DTuner(MeasurementInterface):

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """

        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(IntegerParameter('GPUS', 1, len(cuda.gpus)))

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

        make_program = f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -g -O2 -c {start_path}/stencil2d/CUDAStencilKernel.cu \n'

        if args.parallel:
            make_paralell_start = f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -g -O2 -c -o CUDAStencil.o {start_path}/stencil2d/CUDAStencil.cpp \n'
            make_paralell_start += f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -g -O2 -c -o CommonCUDAStencilFactory.o {start_path}/stencil2d/CommonCUDAStencilFactory.cpp \n'.format(start_path)
            make_paralell_start += f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -g -O2 -c -o Stencil2Dmain.o {start_path}/stencil2d/Stencil2Dmain.cpp \n'.format(start_path)
            make_paralell_start += f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -g -O2 -c -o CUDAStencilFactory.o {start_path}/stencil2d/CUDAStencilFactory.cpp \n'.format(start_path)
            make_paralell_start += f'mpicxx -I {start_path}/common/ -I {start_path}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -g -O2 -c -o main.o {start_path}/cuda-common/main.cpp \n'.format(start_path)
            make_paralell_end = f'mpicxx -L {start_path}/cuda-common -L {start_path}/common -o Stencil2D CUDAStencil.o CommonCUDAStencilFactory.o Stencil2Dmain.o CUDAStencilFactory.o main.o CUDAStencilKernel.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
            compile_cmd = make_paralell_start + make_program + make_paralell_end
        else:
            make_serial_start = f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ -g -O2 -c -o CUDAStencil.o {start_path}/stencil2d/CUDAStencil.cpp \n'
            make_serial_start += f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ -g -O2 -c -o CommonCUDAStencilFactory.o {start_path}/stencil2d/CommonCUDAStencilFactory.cpp \n'
            make_serial_start += f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ -g -O2 -c -o Stencil2Dmain.o {start_path}/stencil2d/Stencil2Dmain.cpp \n'
            make_serial_start += f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ -g -O2 -c -o CUDAStencilFactory.o {start_path}/stencil2d/CUDAStencilFactory.cpp \n'
            make_serial_start += f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ -g -O2 -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_serial_end = f'nvcc -L {start_path}/cuda-common -L {start_path}/common -o Stencil2D CUDAStencil.o CommonCUDAStencilFactory.o Stencil2Dmain.o CUDAStencilFactory.o main.o CUDAStencilKernel.o -lSHOCCommon'
            compile_cmd = make_serial_start + make_program + make_serial_end

        # TODO: remove this
        print(compile_cmd)

        compile_result = self.call_program(compile_cmd)
        assert compile_result['returncode'] == 0

        program_command = './Stencil2D -s ' + str(args.size)

        chosen_gpu_number = cfg['GPUS']
        
        if args.parallel:
            # Select number below max connected GPUs
      
            devices = ','.join([str(i) for i in range(0, chosen_gpu_number)])
            run_cmd = f'mpirun -np {chosen_gpu_number} --allow-run-as-root {program_command} -d {devices}'
        else:
            run_cmd = program_command

        # TODO: remove this
        print(run_cmd)
        run_result = self.call_program(run_cmd)

        # Check that error code and error output is ok
        assert run_result['stderr'] == b''
        assert run_result['returncode'] == 0

        return Result(time=run_result['time'])

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal parameter values written to results.json:", configuration.data)
        
        # Update configuration with problem size
        configuration.data["PROBLEM_SIZE"] = argparser.parse_args().size
        
        self.manipulator().save_to_file(configuration.data, 'results.json')


if __name__ == '__main__':
  argparser = opentuner.default_argparser()
  argparser.add_argument('--size', type=int, default=1, help='problem size of the program (1-4)')
  argparser.add_argument('--gpu-num', type=int, default=1, help='number of GPUs')
  argparser.add_argument('--parallel', type=bool, default=False, help='run on multiple GPUs')
  Stencil2DTuner.main(argparser.parse_args())
