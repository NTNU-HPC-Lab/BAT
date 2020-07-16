import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import json


# Problem sizes used in the SHOC benchmark
problem_sizes = [16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]

gpu = cuda.get_current_device()
max_block_size = gpu.MAX_THREADS_PER_BLOCK

# TODO use different problem sizes from input
size = problem_sizes[8]

# Use host code in combination with CUDA kernel
kernel_files = ['triad_host.cu', '../../../src/kernels/triad/triad_kernel.cu']

# Add parameters to tune
tune_params = dict()
# Using 2^i values less than `gpu.MAX_THREADS_PER_BLOCK`
# TODO: fix program code to be usable for all sizes:
# TODO: [i for i in range(1, gpu.MAX_THREADS_PER_BLOCK + 1)]
tune_params["BLOCK_SIZE"] = list(filter(lambda x: x <= max_block_size, [2**i for i in range(0, 11)]))

tuning_results = tune_kernel("triad_host", kernel_files, size, [], tune_params, lang="C", block_size_names=["BLOCK_SIZE"], compiler_options=["-I ../../../src/kernels/triad/"])

# Save the results as a JSON file
with open("triad-results.json", 'w') as f:
    json.dump(tuning_results, f)
