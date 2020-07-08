import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import math
from verifier import verify_multiple_answers


"""
Input:
const float * __restrict__ valSP,
const double * __restrict__ valDP,
const int    * __restrict__ cols,
const int    * __restrict__ rowDelimiters,
const int dim, 
float * __restrict__ outSP,
double * __restrict__ outDP
"""

with open('../../../src/kernels/spmv/spmv_kernel.cu','r') as f:
    kernel_string = f.read()

gpu = cuda.get_current_device()
sizeIndex = 1
problem_sizes = [1024, 8192, 12288, 16384]
ellpackr_padded_sizes = [1024, 8192, 12288, 16384]
size = problem_sizes[sizeIndex - 1]

with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_csr_normal_val','r') as f:
    values_csr = list(map(numpy.single, f.read().split('\n')))
    
valSP_csr = numpy.array(values_csr)
valDP_csr = numpy.array(list(map(numpy.double, values_csr)))

with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_ellpackr_val','r') as f:
    values_ellpackr = list(map(numpy.single, f.read().split('\n')))
    
valSP_ellpackr = numpy.array(values_ellpackr)
valDP_ellpackr = numpy.array(list(map(numpy.double, values_ellpackr)))

with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_ellpackr_cols','r') as f:
    colVal_csr = list(map(numpy.intc, f.read().split('\n')))

cols_csr = numpy.array(colVal_csr)


with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_ellpackr_cols','r') as f:
    colVal_ellpackr = list(map(numpy.intc, f.read().split('\n')))

cols_ellpackr = numpy.array(colVal_ellpackr)

with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_csr_normal_rowdelimiters','r') as f:
    rowde = list(map(numpy.intc, f.read().split('\n')))

rowDelimiters = numpy.array(rowde)

with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_ellpackr_rowlengths','r') as f:
    rowle = list(map(numpy.intc, f.read().split('\n')))

rowLengths = numpy.array(rowle)

dim = numpy.intc(size)

outSP_csr = numpy.zeros((size), dtype=numpy.single)
outDP_csr = numpy.zeros((size), dtype=numpy.double)

outSP_ellpackr = numpy.zeros(ellpackr_padded_sizes[sizeIndex-1], dtype=numpy.single)
outDP_ellpackr = numpy.zeros(ellpackr_padded_sizes[sizeIndex-1], dtype=numpy.double)


args = [valSP_csr, valDP_csr, valSP_ellpackr, valDP_ellpackr, cols_csr, cols_ellpackr, rowDelimiters, rowLengths, dim, outSP_csr, outDP_csr, outSP_ellpackr, outDP_ellpackr]

min_block_size = 1
max_block_size = gpu.MAX_THREADS_PER_BLOCK

tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(1, max_block_size+1)]
tune_params["PRECISION"] = [32, 64] #, 32]
tune_params["FORMAT"] = [0, 1, 2, 3, 4] # 0: ellpackr, 1: csr-normal-scalar, 2:  csr-padded-scalar, 3: csr-normal-vector, 4: csr-padded-vector
tune_params["UNROLL_LOOP_1"] = [0, 1]#, 1]
tune_params["UNROLL_LOOP_2"] = [0, 1]#, 1]

restrict = ["FORMAT < 3 or (BLOCK_SIZE == 32) or (BLOCK_SIZE == 64) or (BLOCK_SIZE == 128) or (BLOCK_SIZE == 256) or (BLOCK_SIZE == 512) or (BLOCK_SIZE == 1024)", "FORMAT > 2 or (UNROLL_LOOP_2 < 1)"]
#restrict = ["FORMAT < 3 or (BLOCK_SIZE % 32 == 0)", "FORMAT > 2 or (UNROLL_LOOP_2 < 1)"]


with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_csr_normal_vec','r') as f:
    tex_read = list(map(numpy.single, f.read().split('\n')))

tex = numpy.array(tex_read).reshape(1,size)

params_sp_scalar = { "BLOCK_SIZE": 128, "PRECISION": 32, "FORMAT": 1, "UNROLL_LOOP": 0, "UNROLL_LOOP_2": 0 }
results_sp_scalar = run_kernel("spmv_kernel", kernel_string, size, args, params_sp_scalar, 
    texmem_args={ 'vecTex': { 'array': tex, 'address_mode': 'clamp' } }, 
    block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])

params_dp_scalar = { "BLOCK_SIZE": 128, "PRECISION": 64, "FORMAT": 1, "UNROLL_LOOP": 0, "UNROLL_LOOP_2": 0 }
results_dp_scalar = run_kernel("spmv_kernel", kernel_string, size, args, params_dp_scalar, 
    texmem_args={ 'vecTex': { 'array': tex, 'address_mode': 'clamp' } }, 
    block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])

params_sp_vector = { "BLOCK_SIZE": 128, "PRECISION": 32, "FORMAT": 3, "UNROLL_LOOP": 0, "UNROLL_LOOP_2": 0 }
results_sp_vector = run_kernel("spmv_kernel", kernel_string, size, args, params_sp_vector,
    smem_args={'size': 128 },
    texmem_args={ 'vecTex': { 'array': tex, 'address_mode': 'clamp' } }, 
    block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])

params_dp_vector = { "BLOCK_SIZE": 128, "PRECISION": 64, "FORMAT": 3, "UNROLL_LOOP": 0, "UNROLL_LOOP_2": 0 }
results_dp_vector = run_kernel("spmv_kernel", kernel_string, size, args, params_dp_vector, 
    smem_args={'size': 128 },
    texmem_args={ 'vecTex': { 'array': tex, 'address_mode': 'clamp' } }, 
    block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])


params_sp_ellpackr = { "BLOCK_SIZE": 128, "PRECISION": 32, "FORMAT": 0, "UNROLL_LOOP": 0, "UNROLL_LOOP_2": 0 }
results_sp_ellpackr = run_kernel("spmv_kernel", kernel_string, size, args, params_sp_ellpackr,
    smem_args={'size': 128 },
    texmem_args={ 'vecTex': { 'array': tex, 'address_mode': 'clamp' } }, 
    block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])

params_dp_ellpackr = { "BLOCK_SIZE": 128, "PRECISION": 64, "FORMAT": 0, "UNROLL_LOOP": 0, "UNROLL_LOOP_2": 0 }
results_dp_ellpackr = run_kernel("spmv_kernel", kernel_string, size, args, params_dp_ellpackr,
    smem_args={'size': 128 },
    texmem_args={ 'vecTex': { 'array': tex, 'address_mode': 'clamp' } }, 
    block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])


answer_sp_scalar = [None, None, None, None, None, None, None, None, None, results_sp_scalar[9], None, None, None]
answer_dp_scalar = [None, None, None, None, None, None, None, None, None, None, results_dp_scalar[10], None, None]
answer_sp_vector = [None, None, None, None, None, None, None, None, None, results_sp_vector[9], None, None, None]
answer_dp_vector = [None, None, None, None, None, None, None, None, None, None, results_dp_vector[10], None, None]
answer_sp_ellpackr = [None, None, None, None, None, None, None, None, None, None, None, results_sp_ellpackr[11], None]
answer_dp_ellpackr = [None, None, None, None, None, None, None, None, None, None, None, None, results_dp_ellpackr[12]]


answers = [answer_sp_scalar, answer_dp_scalar, answer_sp_vector, answer_dp_vector, answer_sp_ellpackr, answer_dp_ellpackr]

tune_kernel("spmv_kernel", kernel_string, size, args, tune_params, 
    texmem_args={ 'vecTex': { 'array': tex, 'address_mode': 'clamp' } }, 
    block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"], 
    restrictions=restrict, answer=answers, verify=verify_multiple_answers)


# tuning_results = 
# import json
# with open("results.json", 'w') as fp:
#    json.dump(tuning_results[0], fp)
