import json
import time
import cupy as cp

from src.reader.T1_specification import populate_args, get_launch_config
from src.benchmarks.correctness import correctness_funcs
from src.reader.util import get_kernel_path

DEBUG = False


def launch_kernel(args_tuple, launch_config, kernel):
    grid_dim = (launch_config["GRID_SIZE_X"], launch_config["GRID_SIZE_Y"], launch_config["GRID_SIZE_Z"])
    block_dim = (launch_config["BLOCK_SIZE_X"], launch_config["BLOCK_SIZE_Y"], launch_config["BLOCK_SIZE_Z"])
    shared_mem_size = launch_config.get("SHARED_MEMORY_SIZE", 0)
    if DEBUG:
        print(grid_dim, block_dim)
        print("Before")
        for arg in args_tuple:
            print(arg, arg.size)
    t0 = time.time()
    for i in range(launch_config.get("ITERATIONS", 10)):
        kernel(grid=grid_dim, block=block_dim, args=args_tuple, shared_mem=shared_mem_size)
    duration = time.time() - t0
    if DEBUG:
        print("After")
        for arg in args_tuple:
            print(arg, arg.size)
    return duration


def get_kernel(kernel_spec, compiler_options):
    kernel_name = kernel_spec["kernelName"]
    jitify = kernel_spec.get("jitify", False)
    with open(get_kernel_path(kernel_spec), 'r') as f:
        module = cp.RawModule(code=f.read(), name_expressions=[],
                              options=tuple(compiler_options), jitify=jitify)
    return module.get_function(kernel_name)


def generate_compiler_options(kernel_spec, tuning_config, benchmark_config):
    compiler_options = kernel_spec["compilerOptions"]
    for (key, val) in tuning_config.items():
        compiler_options.append("-D{}={}".format(key, val))
    for (key, val) in benchmark_config.items():
        compiler_options.append("-D{}={}".format(key, val))
    return compiler_options


def run_kernel(kernel_spec, launch_config, tuning_config, benchmark_config):
    compiler_options = generate_compiler_options(kernel_spec,
                                                 tuning_config, benchmark_config)
    args = populate_args(kernel_spec)
    lf_ker = get_kernel(kernel_spec, compiler_options)
    args_tuple = tuple(args)
    result = launch_kernel(args_tuple, launch_config, lf_ker)
    correctness = correctness_funcs[kernel_spec["kernelName"]]
    correctness(tuple(args), args_tuple, tuning_config, launch_config)
    return result


def core(json_path, benchmark_config, tuning_config, testing):
    with open(json_path, 'r') as f:
        r = json.load(f)

    kernel_spec = r["kernelSpecification"]
    launch_config = get_launch_config(kernel_spec, tuning_config)

    return run_kernel(kernel_spec, launch_config, tuning_config, benchmark_config)
