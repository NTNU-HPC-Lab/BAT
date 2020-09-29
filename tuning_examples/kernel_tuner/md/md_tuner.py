import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import json
import argparse

# Setup CLI parser
parser = argparse.ArgumentParser(description="Reduction tuner")
parser.add_argument("--size", "-s", type=int, default=1, help="problem size to the benchmark (e.g.: 2)")
arguments = parser.parse_args()

# Problem sizes used in the SHOC benchmark
problem_sizes = [12288, 24576, 36864, 73728]

gpu = cuda.get_current_device()
max_block_size = gpu.MAX_THREADS_PER_BLOCK

input_problem_size = arguments.size
size = problem_sizes[input_problem_size - 1]

# Use host code in combination with CUDA kernel
kernel_files = ['md_host.cu', '../../../src/kernels/md/md_kernel.cu']

tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(1, max_block_size + 1)] # Range: [1, ..., max_block_size]
tune_params["PRECISION"] = [32, 64]
tune_params["TEXTURE_MEMORY"] = [0, 1]
tune_params["WORK_PER_THREAD"] = [i for i in range(1, 6)] # Range: [1, ..., 5]

# Tune all kernels and correctness verify by throwing error if verification failed
tuning_results = tune_kernel("md_host", kernel_files, size, [], tune_params, lang="C", block_size_names=["BLOCK_SIZE"], compiler_options=["-I ../../../src/kernels/md/", f"-DPROBLEM_SIZE={input_problem_size}"])

# Save the results as a JSON file
with open("md-results.json", 'w') as f:
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
best_parameters["PROBLEM_SIZE"] = input_problem_size

# Save the best results as a JSON file
with open("best-md-results.json", 'w') as f:
    json.dump(best_parameters, f)
