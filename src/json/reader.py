import json
import time
from kernel_specification import populate_args, get_launch_config

import cupy as cp


def print_json(j):
    print(json.dumps(j, indent=4, sort_keys=True))


def launch_kernel(args_tuple, launch_config, kernel):
    grid_dim = (launch_config["GRID_SIZE_X"], launch_config["GRID_SIZE_Y"], launch_config["GRID_SIZE_Z"])
    block_dim = (launch_config["BLOCK_SIZE_X"], launch_config["BLOCK_SIZE_Y"], launch_config["BLOCK_SIZE_Z"])
    print(grid_dim, block_dim)

    print(args_tuple[1], len(args_tuple[1]))
    t0 = time.time()
    kernel(grid=grid_dim, block=block_dim, args=args_tuple)
    duration = time.time() - t0
    print(args_tuple[0], len(args_tuple[0]))
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
    return compiler_options


def run_kernel(kernel_spec, launch_config, tuning_config, benchmark_config):
    compiler_options = generate_compiler_options(kernel_spec,
                                                 tuning_config, benchmark_config)
    args = populate_args(kernel_spec)
    lf_ker = get_kernel(kernel_spec, compiler_options)
    return launch_kernel(tuple(args), launch_config, lf_ker)


def core(json_path, benchmark_config, tuning_config):
    with open(json_path, 'r') as f:
        r = json.load(f)

    kernel_spec = r["kernelSpecification"]
    launch_config = get_launch_config(kernel_spec, tuning_config)

    return run_kernel(kernel_spec, launch_config, tuning_config, benchmark_config)
