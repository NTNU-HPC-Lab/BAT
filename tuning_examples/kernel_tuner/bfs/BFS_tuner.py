import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda

with open('../../../src/kernels/bfs/BFS_kernel.cu','r') as f:
    kernel_string = f.read()

gpu = cuda.get_current_device()
vertices = 1000000

# unsigned int *levels, length=numVerts, value=0 on first place, 4294967295 on rest
a = numpy.full(shape=(vertices), fill_value=4294967295, dtype=numpy.uintc)
a[0] = 0
# unsigned int *edgeArray, length=vertices+1, value=edgearray file
with open('../../../src/kernels/bfs/data/' + str(vertices) + 'edgearray','r') as f:
    output = list(map(numpy.uintc, f.read().split('\n')))
b = numpy.array(output)

# unsigned int *edgeArrayAux, length=vertices*2-2, value=edgearrayaux file
with open('../../../src/kernels/bfs/data/' + str(vertices) + 'edgearrayaux','r') as f:
    outputaux = list(map(numpy.uintc, f.read().split('\n')))
c = numpy.array(outputaux)

# int W_SZ, value=32
d = numpy.intc(32)
# unsigned int numVertices, value=vertices
e = numpy.uintc(vertices)
# int curr, value=0
f = numpy.intc(0)
# int *flag, value=0
g = numpy.array([0], numpy.intc)

args = [a, b, c, d, e, f, g]

min_block_size = 1
max_block_size = min(vertices, gpu.MAX_THREADS_PER_BLOCK)

tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(min_block_size, max_block_size+1)]
tune_params["CHUNK_SIZE"] = [32, 64, 128, 256]
tune_params["UNROLL_OUTER_LOOP"] = [0, 1]
tune_params["UNROLL_INNER_LOOP"] = [0, 1]


params = { "BLOCK_SIZE": 1024, "CHUNK_SIZE": 32, "UNROLL_OUTER_LOOP": 0, "UNROLL_INNER_LOOP": 0 }
results = run_kernel("BFS_kernel_warp", kernel_string, vertices, args, params, block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])

#set non-output fields to None
answer = [results[0], None, None, None, None, None, results[6]]

tune_kernel("BFS_kernel_warp", kernel_string, vertices, args, tune_params, block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"], answer=answer)