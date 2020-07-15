import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import json


# Load kernel from file
with open('../../../src/kernels/triad/triad_kernel.cu', 'r') as f:
    kernel_string = f.read()

# Problem sizes used in the SHOC benchmark
problem_sizes = [16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]

gpu = cuda.get_current_device()
max_block_size = gpu.MAX_THREADS_PER_BLOCK

# TODO use different problem sizes from input
size = problem_sizes[8]
multiplier = 10.0

# Add arguments for kernel
# Arrays (A, B, C) have random numbers similar to the program. The numbers are in the range [0, 10)
A = multiplier * numpy.random.rand(size).astype(numpy.single)
B = multiplier * numpy.random.rand(size).astype(numpy.single)
C = multiplier * numpy.random.rand(size).astype(numpy.single)
s = numpy.single(1.75) # = scalar

args = [A, B, C, s]

tune_params = dict()
# Using 2^i values less than `gpu.MAX_THREADS_PER_BLOCK`
# TODO: fix program code to be usable for all sizes:
# TODO: [i for i in range(1, gpu.MAX_THREADS_PER_BLOCK + 1)]
tune_params["BLOCK_SIZE"] = list(filter(lambda x: x <= max_block_size, [2**i for i in range(0, 11)]))

# Run kernel with known working parameters for correctness verification
params = { "BLOCK_SIZE": 16 }
results = run_kernel("triad", kernel_string, size, args, params, block_size_names=["BLOCK_SIZE"])

# Set non-output fields to None
answer = [None, None, results[2], None]

tuning_results = tune_kernel("triad", kernel_string, size, args, tune_params, answer=answer, block_size_names=["BLOCK_SIZE"])

# Save the results as a JSON file
with open("triad-results.json", 'w') as f:
    json.dump(tuning_results, f)
