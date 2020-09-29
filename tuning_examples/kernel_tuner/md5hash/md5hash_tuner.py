import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import json
import argparse

# Setup CLI parser
parser = argparse.ArgumentParser(description="Reduction tuner")
parser.add_argument("--size", "-s", type=int, default=1, help="problem size to the benchmark (e.g.: 2)")
arguments = parser.parse_args()

size = arguments.size
gpu = cuda.get_current_device()
byte_lengths = [7, 5, 6, 5]
byte_length = byte_lengths[size - 1]

# Use host code in combination with CUDA kernel
kernel_files = ['md5hash_host.cu', '../../../src/kernels/md5hash/md5hash_kernel.cu']

min_block_size = 1
max_block_size = gpu.MAX_THREADS_PER_BLOCK

tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(min_block_size, max_block_size+1)]
tune_params["ROUND_STYLE"] = [0, 1]
tune_params["UNROLL_LOOP_1"] = [0, 1] # 1 after name so Kernel Tuner wont replace TEXTURE_MEMORY_EA in TEXTURE_MEMORY_EAA with value
tune_params["UNROLL_LOOP_2"] = [0, 1]
tune_params["UNROLL_LOOP_3"] = [0, 1]
tune_params["INLINE_1"] = [0, 1]
tune_params["INLINE_2"] = [0, 1]
tune_params["WORK_PER_THREAD_FACTOR"] = [1, 2, 3, 4, 5]


tuning_results = tune_kernel("RunBenchmark", kernel_files, byte_length, [], tune_params, lang="C", block_size_names=["BLOCK_SIZE"], 
    compiler_options=["-I ../../../src/kernels/md5hash/", "-I ../../../src/programs/common/", "-I ../../../src/programs/cuda-common/", f"-DPROBLEM_SIZE={size}"])


# Save the results as a JSON file
with open("md5hash-results.json", 'w') as f:
    json.dump(tuning_results, f)

# Get the best configuration
best_parameter_config = min(tuning_results[0], key=lambda x: x['time'])
best_parameters = dict()

# Filter out parameters from results
for k, v in best_parameter_config.items():
    if k not in tune_params:
        continue

    best_parameters[k] = v

# Add problem size to results
best_parameters["PROBLEM_SIZE"] = size

# Save the best results as a JSON file
with open("best-md5hash-results.json", 'w') as f:
    json.dump(best_parameters, f)
