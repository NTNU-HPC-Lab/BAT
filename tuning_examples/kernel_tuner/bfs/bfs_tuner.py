import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import json
import argparse
from numpyencoder import NumpyEncoder

# Setup CLI parser
parser = argparse.ArgumentParser(description="BFS tuner")
parser.add_argument("--size", "-s", type=int, default=1, help="problem size to the benchmark (e.g.: 2)")
parser.add_argument("--technique", "-t", type=str, default="brute_force", help="tuning technique to use for the benchmark (e.g.: annealing)")
arguments = parser.parse_args()

size = arguments.size
gpu = cuda.get_current_device()
sizes = [1000, 10000, 100000, 1000000]
vertices = sizes[size - 1]

# Use host code in combination with CUDA kernel
kernel_files = ['bfs_host.cu', '../../../src/programs/bfs/bfs_kernel.cu']

min_block_size = 1
max_block_size = min(vertices, gpu.MAX_THREADS_PER_BLOCK)

tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(min_block_size, max_block_size+1)]
tune_params["CHUNK_FACTOR"] = [1, 2, 4, 8]
tune_params["TEXTURE_MEMORY_EA1"] = [0, 1] # 1 after name so Kernel Tuner wont replace TEXTURE_MEMORY_EA in TEXTURE_MEMORY_EAA with value
tune_params["TEXTURE_MEMORY_EAA"] = [0, 1]
tune_params["UNROLL_OUTER_LOOP"] = [0, 1]
tune_params["UNROLL_INNER_LOOP"] = [0, 1]

strategy_options = {}
if arguments.technique == "genetic_algorithm":
    strategy_options = {"maxiter": 50, "popsize": 10}

tuning_results = tune_kernel("RunBenchmark", kernel_files, vertices, [], tune_params, strategy=arguments.technique, lang="C", block_size_names=["BLOCK_SIZE"], 
    compiler_options=["-I ../../../src/programs/bfs/", "-I ../../../src/programs/common/", "-I ../../../src/programs/cuda-common/", f"-DPROBLEM_SIZE={size}"],
    iterations=2, strategy_options=strategy_options)


# Save the results as a JSON file
with open("bfs-results.json", 'w') as f:
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
with open("best-bfs-results.json", 'w') as f:
    json.dump(best_parameters, f, indent=4, cls=NumpyEncoder)
