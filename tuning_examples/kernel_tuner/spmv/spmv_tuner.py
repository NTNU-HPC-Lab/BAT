import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import math


with open('../../../src/kernels/spmv/spmv_kernel.cu','r') as f:
    kernel_string = f.read()

gpu = cuda.get_current_device()
sizeIndex = 1
problem_sizes = [1024, 8192, 12288, 16384]
size = problem_sizes[sizeIndex - 1]

with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_csr_scalar_normal_val','r') as f:
    values = list(map(numpy.single, f.read().split('\n')))
    
valSP = numpy.array(values)
valDP = numpy.array(list(map(numpy.double, values)))

with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_csr_scalar_normal_cols','r') as f:
    colVal = list(map(numpy.intc, f.read().split('\n')))

cols = numpy.array(colVal)

with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_csr_scalar_normal_rowdelimiters','r') as f:
    rowde = list(map(numpy.intc, f.read().split('\n')))

rowDelimiters = numpy.array(rowde)

dim = numpy.intc(size)

outSP = numpy.zeros((size), dtype=numpy.single)

outDP = numpy.zeros((size), dtype=numpy.double)

"""
const float * __restrict__ valSP,
const double * __restrict__ valDP,
const int    * __restrict__ cols,
const int    * __restrict__ rowDelimiters,
const int dim, 
float * __restrict__ outSP,
double * __restrict__ outDP
"""


args = [valSP, valDP, cols, rowDelimiters, dim, outSP, outDP]

min_block_size = 1
max_block_size = gpu.MAX_THREADS_PER_BLOCK

tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(min_block_size, max_block_size+1)]
tune_params["PRECISION"] = [64, 32]#, 64]
tune_params["UNROLL_LOOP"] = [0, 1]

with open('../../../src/kernels/spmv/data/' + str(sizeIndex) + '_csr_scalar_normal_vec','r') as f:
    xx = list(map(numpy.single, f.read().split('\n')))

x = numpy.array(xx).reshape(1,size)

def verify_multiple_answers(answer, gpu_result, atol=None):
    def _ravel(a):
        if hasattr(a, 'ravel') and len(a.shape) > 1:
            return a.ravel()
        return a

    def _flatten(a):
        if hasattr(a, 'flatten'):
            return a.flatten()
        return a

    def check_similarity(ans, gpu_res):
        for i in range(0, len(ans)): 
            expected = ans[i]
            if expected is not None:
                result = _ravel(gpu_res[i])
                expected = _flatten(expected)
                return numpy.allclose(expected, result, atol=atol)
        return False # This will fail if every value is None, but no verification is needed in that case

    passing = False
    if isinstance(answer[0], list):
        for i in range(0, len(answer)):
            res = check_similarity(answer[i], gpu_result)
            if res is True:
                passing = True
        return passing
    else:
        print('rrrrrrrrrrrrr')
        res = check_similarity(answer, gpu_result)
        if res is False:
            print('Answers does not match')
        return res
                
    return True



params_sp = { "BLOCK_SIZE": 128, "PRECISION": 32, "UNROLL_LOOP": 0 }
results_sp = run_kernel("spmv_csr_scalar_kernel", kernel_string, size, args, params_sp, texmem_args={ 'vecTex': { 'array': x, 'address_mode': 'clamp' } }, block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])
params_dp = { "BLOCK_SIZE": 128, "PRECISION": 64, "UNROLL_LOOP": 0 }
results_dp = run_kernel("spmv_csr_scalar_kernel", kernel_string, size, args, params_dp, texmem_args={ 'vecTex': { 'array': x, 'address_mode': 'clamp' } }, block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"])

answer_sp = [None, None, None, None, None, results_sp[5], None]
answer_dp = [None, None, None, None, None, None, results_dp[6]]

answers = [answer_dp, answer_sp]

tune_kernel("spmv_csr_scalar_kernel", kernel_string, size, args, tune_params, texmem_args={ 'vecTex': { 'array': x, 'address_mode': 'clamp' } }, block_size_names=["BLOCK_SIZE", "block_size_y", "block_size_z"], answer=answers, verify=verify_multiple_answers)
