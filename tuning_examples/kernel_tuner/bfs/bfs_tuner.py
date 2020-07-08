import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda

with open('../../../src/kernels/bfs/BFS_kernel.cu','r') as f:
    kernel_string = f.read()

size = 2
gpu = cuda.get_current_device()
sizes = [1000, 10000, 100000, 1000000]

vertices = sizes[size - 1]

# unsigned int *levels, length=vertices, value=0 on first place, 4294967295 on rest
levels = numpy.full(shape=(vertices), fill_value=4294967295, dtype=numpy.uintc)
levels[0] = 0
# unsigned int *edgeArray, length=vertices+1, value=edgearray file
with open('../../../src/kernels/bfs/data/' + str(size) + '_edgearray','r') as f:
    output = list(map(numpy.uintc, f.read().split('\n')))
edgeArray = numpy.array(output)

texEA = numpy.array(output).reshape(1,vertices+1)

# unsigned int *edgeArrayAux, length=vertices*2-2, value=edgearrayaux file
with open('../../../src/kernels/bfs/data/' + str(size) + '_edgearrayaux','r') as f:
    outputaux = list(map(numpy.uintc, f.read().split('\n')))
edgeArrayAux = numpy.array(outputaux)

texEAA = numpy.array(outputaux).reshape(1,vertices*2-2)

# int W_SZ, value=32
W_SZ = numpy.intc(32)
# unsigned int numVertices, value=vertices
numVertices = numpy.uintc(vertices)
# int curr, value=0
curr = numpy.intc(0)
# int *flag, value=0
flag = numpy.array([0], numpy.intc)

args = [levels, edgeArray, edgeArrayAux, W_SZ, numVertices, curr, flag]

min_block_size = 1
max_block_size = min(vertices, gpu.MAX_THREADS_PER_BLOCK)

tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(min_block_size, max_block_size+1)]
tune_params["CHUNK_SIZE"] = [32, 64, 128, 256]
tune_params["TEXTURE_MEMORY_EA1"] = [0, 1] # 1 after name so Kernel Tuner wont replace TEXTURE_MEMORY_EA in TEXTURE_MEMORY_EAA with value
tune_params["TEXTURE_MEMORY_EAA"] = [0, 1]
tune_params["UNROLL_OUTER_LOOP"] = [0, 1]
tune_params["UNROLL_INNER_LOOP"] = [0, 1]


params = { "BLOCK_SIZE": 1024, "CHUNK_SIZE": 32, "TEXTURE_MEMORY_EA1": 0, "TEXTURE_MEMORY_EAA": 0, "UNROLL_OUTER_LOOP": 0, "UNROLL_INNER_LOOP": 0 }
results = run_kernel("BFS_kernel_warp", kernel_string, vertices, args, params,
    texmem_args={ 'textureEA': { 'array': texEA, 'address_mode': 'clamp' }, 'textureEAA': { 'array': texEAA, 'address_mode': 'clamp' } },  
    block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])

#set non-output fields to None
answer = [results[0], None, None, None, None, None, results[6]]

tune_kernel("BFS_kernel_warp", kernel_string, vertices, args, tune_params, 
    texmem_args={ 'textureEA': { 'array': texEA, 'address_mode': 'clamp' }, 'textureEAA': { 'array': texEAA, 'address_mode': 'clamp' } },  
    block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"], answer=answer)