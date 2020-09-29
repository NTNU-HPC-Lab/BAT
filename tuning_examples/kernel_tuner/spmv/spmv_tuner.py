import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import math
import json

gpu = cuda.get_current_device()
sizeIndex = 1
problem_sizes = [1024, 8192, 12288, 16384]
ellpackr_padded_sizes = [1024, 8192, 12288, 16384]
size = problem_sizes[sizeIndex - 1]

# Use host code in combination with CUDA kernel
kernel_files = ['spmv_host.cu', '../../../src/kernels/spmv/spmv_kernel_no_template.cu']

min_block_size = 1
max_block_size = gpu.MAX_THREADS_PER_BLOCK

tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(1, max_block_size+1)]
tune_params["PRECISION"] = [32, 64]
tune_params["FORMAT"] = [0, 1, 2, 3, 4] # 0: ellpackr, 1: csr-normal-scalar, 2:  csr-padded-scalar, 3: csr-normal-vector, 4: csr-padded-vector
tune_params["UNROLL_LOOP_1"] = [0, 1]
tune_params["UNROLL_LOOP_2"] = [0, 1]
tune_params["PROBLEM_SIZE"] = [sizeIndex] # For setting problem size in host code

restrict = ["FORMAT < 3 or (BLOCK_SIZE == 32) or (BLOCK_SIZE == 64) or (BLOCK_SIZE == 128) or (BLOCK_SIZE == 256) or (BLOCK_SIZE == 512) or (BLOCK_SIZE == 1024)", "FORMAT > 2 or (UNROLL_LOOP_2 < 1)"]

tuning_results = tune_kernel("RunBenchmark", kernel_files, size, [], tune_params, restrictions=restrict, lang="C", block_size_names=["BLOCK_SIZE"], 
    compiler_options=["-I ../../../src/kernels/spmv/", "-I ../../../src/programs/common/", "-I ../../../src/programs/cuda-common/"])


# Save the results as a JSON file
with open("spmv-results.json", 'w') as f:
    json.dump(tuning_results, f)

# Get the best configuration
best_parameter_config = min(tuning_results[0], key=lambda x: x['time'])
best_parameters = dict()

# Filter out parameters from results
for k, v in best_parameter_config.items():
    if k not in tune_params:
        continue

    best_parameters[k] = v

# Save the best results as a JSON file
with open("best-spmv-results.json", 'w') as f:
    json.dump(best_parameters, f)
