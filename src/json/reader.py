import json
import sys
import time

from kernel_specification import populate_args, get_launch_config

import cupy as cp
import numpy as np


def print_json(j):
    print(json.dumps(j, indent=4, sort_keys=True))


DEBUG = 0


def launch_kernel(args_tuple, launch_config, kernel):
    grid_dim = (launch_config["GRID_SIZE_X"], launch_config["GRID_SIZE_Y"], launch_config["GRID_SIZE_Z"])
    block_dim = (launch_config["BLOCK_SIZE_X"], launch_config["BLOCK_SIZE_Y"], launch_config["BLOCK_SIZE_Z"])
    if DEBUG:
        print(grid_dim, block_dim)
        print("Before")
        for arg in args_tuple:
            print(arg, arg.size)
    t0 = time.time()
    kernel(grid=grid_dim, block=block_dim, args=args_tuple)
    duration = time.time() - t0
    if DEBUG:
        print("After")
        for arg in args_tuple:
            print(arg, arg.size)
    return duration


def get_kernel(kernel_spec, compiler_options):
    kernel_name = kernel_spec["kernelName"]
    with open(kernel_spec["kernelFile"], 'r') as f:
        kernel_str = f.read()
        module = cp.RawModule(code=kernel_str, name_expressions=[kernel_name],
                              options=tuple(compiler_options))
    return module.get_function(kernel_name)


def generate_compiler_options(kernel_spec, tuning_config, benchmark_config):
    compiler_options = kernel_spec["compilerOptions"]
    for (key, val) in tuning_config.items():
        compiler_options.append("-D{}={}".format(key, val))
    for (key, val) in benchmark_config.items():
        compiler_options.append("-D{}={}".format(key, val))
    # print(compiler_options)
    return compiler_options

def md_correctness(args_before, args_after, config):
    print(config)
    print(args_before)
    print(args_after)

def builtin_vectors_correctness(args_before, args_after, config):
    left = args_after[2][0].item()[0]
    right = args_before[0][0].item()[0] + args_before[1][0].item()[0]
    if left == 0:
        print("Failed to initialize",  args_after[3])
    elif left != right:
        print("Did not pass", left, "!=", right, args_after[3], config)
        exit(1)
    else:
        print("Passed", left, right, args_after[3], config)

def md5hash_correctness(args_before, args_after, config):
    print("MD5Hash Correctness")
    print(config)
    print(args_before)
    print(args_after)


correctness_funcs = {
    "sum_kernel": builtin_vectors_correctness,
    "compute_lj_force": md_correctness,
    "FindKeyWithDigest_Kernel": md5hash_correctness, 
}


def run_kernel(kernel_spec, launch_config, tuning_config, benchmark_config):
    compiler_options = generate_compiler_options(kernel_spec,
                                                 tuning_config, benchmark_config)
    # compiler_options = ['-std=c++11', '-DBLOCK_SIZE=1024']
    args = populate_args(kernel_spec)
    lf_ker = get_kernel(kernel_spec, compiler_options)
    args_tuple = tuple(args)
    # launch_config = {'GRID_SIZE_X': 4, 'GRID_SIZE_Y': 1, 'GRID_SIZE_Z': 1, 'BLOCK_SIZE_X': 1024, 'BLOCK_SIZE_Y': 1, 'BLOCK_SIZE_Z': 1}
    # launch_config = {'GRID_SIZE_X': 5, 'GRID_SIZE_Y': 1, 'GRID_SIZE_Z': 1, 'BLOCK_SIZE_X': 928, 'BLOCK_SIZE_Y': 1, 'BLOCK_SIZE_Z': 1}
    # launch_config = {'GRID_SIZE_X': 16, 'GRID_SIZE_Y': 1, 'GRID_SIZE_Z': 1, 'BLOCK_SIZE_X': 256, 'BLOCK_SIZE_Y': 1,
    #                 'BLOCK_SIZE_Z': 1}
    print(launch_config, args_tuple)
    result = launch_kernel(args_tuple, launch_config, lf_ker)
    correctness = correctness_funcs[kernel_spec["kernelName"]]
    correctness(tuple(args), args_tuple, launch_config)
    del args
    cp._default_memory_pool.free_all_free()
    return result


def core(json_path, benchmark_config, tuning_config):
    with open(json_path, 'r') as f:
        r = json.load(f)

    kernel_spec = r["kernelSpecification"]
    launch_config = get_launch_config(kernel_spec, tuning_config)

    return run_kernel(kernel_spec, launch_config, tuning_config, benchmark_config)
