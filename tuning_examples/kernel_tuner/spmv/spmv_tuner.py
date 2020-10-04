import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import math
import json
import argparse
from numpyencoder import NumpyEncoder

# Setup CLI parser
parser = argparse.ArgumentParser(description="SPMV tuner")
parser.add_argument("--size", "-s", type=int, default=1, help="problem size to the benchmark (e.g.: 2)")
parser.add_argument("--technique", "-t", type=str, default="brute_force", help="tuning technique to use for the benchmark (e.g.: annealing)")
arguments = parser.parse_args()

gpu = cuda.get_current_device()
sizeIndex = arguments.size
problem_sizes = [1024, 8192, 12288, 16384]
size = problem_sizes[sizeIndex - 1]

# Use host code in combination with CUDA kernel
kernel_files = ['spmv_host.cu', '../../../src/kernels/spmv/spmv_kernel_no_template.cu']

min_block_size = 1
max_block_size = gpu.MAX_THREADS_PER_BLOCK

tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(min_block_size, max_block_size+1)]
tune_params["PRECISION"] = [32, 64]
tune_params["FORMAT"] = [0, 1, 2, 3, 4] # 0: ellpackr, 1: csr-normal-scalar, 2:  csr-padded-scalar, 3: csr-normal-vector, 4: csr-padded-vector
tune_params["UNROLL_LOOP_1"] = [0, 1]
tune_params["UNROLL_LOOP_2"] = [0, 1]
tune_params["TEXTURE_MEMORY"] = [0, 1]

restrict = ["FORMAT < 3 or BLOCK_SIZE % 32 == 0", "FORMAT > 2 or (UNROLL_LOOP_2 < 1)"]

tuning_results = tune_kernel("RunBenchmark", kernel_files, size, [], tune_params, strategy=arguments.technique, restrictions=restrict, lang="C", block_size_names=["BLOCK_SIZE"], 
    compiler_options=["-I ../../../src/kernels/spmv/", "-I ../../../src/programs/common/", "-I ../../../src/programs/cuda-common/", f"-DPROBLEM_SIZE={sizeIndex}"])


# Save the results as a JSON file
with open("spmv-results.json", 'w') as f:
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
best_parameters["PROBLEM_SIZE"] = sizeIndex
best_parameters["TUNING_TECHNIQUE"] = arguments.technique

# Save the best results as a JSON file
with open("best-spmv-results.json", 'w') as f:
    json.dump(best_parameters, f, indent=4, cls=NumpyEncoder)
