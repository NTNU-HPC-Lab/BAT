import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import json
import argparse
from numpyencoder import NumpyEncoder

# Setup CLI parser
parser = argparse.ArgumentParser(description="Scan tuner")
parser.add_argument("--size", "-s", type=int, default=1, help="problem size to the benchmark (e.g.: 2)")
parser.add_argument("--technique", "-t", type=str, default="brute_force", help="tuning technique to use for the benchmark (e.g.: annealing)")
arguments = parser.parse_args()

size = arguments.size
problem_sizes = [1, 8, 32, 64]
problem_size = problem_sizes[size - 1]

# Use host code in combination with CUDA kernel
kernel_files = ['scan_host.cu', '../../../src/kernels/scan/scan_kernel_no_template.cu']

block_sizes = [2**i for i in range(4, 10)]
block_sizes.remove(32)
grid_sizes = [2**i for i in range(0, 10)]

tune_params = dict()
tune_params["BLOCK_SIZE"] = block_sizes
tune_params["GRID_SIZE"] = grid_sizes
tune_params["PRECISION"] = [32, 64]
tune_params["UNROLL_LOOP_1"] = [0, 1]
tune_params["UNROLL_LOOP_2"] = [0, 1]

restrict = ["GRID_SIZE <= BLOCK_SIZE"]

tuning_results = tune_kernel("RunBenchmark", kernel_files, problem_size, [], tune_params, strategy=arguments.technique, restrictions=restrict, lang="C", block_size_names=["BLOCK_SIZE"], 
    compiler_options=["-I ../../../src/kernels/scan/", "-I ../../../src/programs/common/", "-I ../../../src/programs/cuda-common/", f"-DPROBLEM_SIZE={size}", "-DNOT_USING_HELPERS=1"])

# Save the results as a JSON file
with open("scan-results.json", 'w') as f:
    json.dump(tuning_results, f, indent=4, cls=NumpyEncoder)

# Get the best configuration
best_parameter_config = min(tuning_results[0], key=lambda x: x['time'])
best_parameters = dict()

# Filter out parameters from results
for k, v in best_parameter_config.items():
    if k not in tune_params:
        continue

    best_parameters[k] = v

# Add problem size and tuning technique to results
best_parameters["PROBLEM_SIZE"] = size
best_parameters["TUNING_TECHNIQUE"] = arguments.technique

# Save the best results as a JSON file
with open("best-scan-results.json", 'w') as f:
    json.dump(best_parameters, f, indent=4, cls=NumpyEncoder)
