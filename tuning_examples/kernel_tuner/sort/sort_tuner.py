import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import pycuda.driver as drv
import json


# Problem sizes used in the SHOC benchmark
problem_sizes = [1, 8, 48, 96]
input_problem_size = 4
size = int((problem_sizes[input_problem_size - 1] * 1024 * 1024) / 4) # 4 = sizeof(uint)

# Use host code in combination with CUDA kernel
kernel_files = ['sort_host.cu', '../../../src/kernels/sort/sort_kernel.cu']

# Add parameters to tune
tune_params = dict()
tune_params["LOOP_UNROLL_LSB"] = [0, 1]
tune_params["LOOP_UNROLL_LOCAL_MEMORY"] = [0, 1]
tune_params["LOOP_UNROLL_ADD_UNIFORM"] = [0, 1]

# Tune all kernels and correctness verify by throwing error if verification failed
tuning_results = tune_kernel("sort", kernel_files, size, [], tune_params, lang="C", compiler_options=["-I ../../../src/kernels/sort/"])

# Save the results as a JSON file
with open("sort-results.json", 'w') as f:
    json.dump(tuning_results, f)
