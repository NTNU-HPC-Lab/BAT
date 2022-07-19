import json
import random
import time

import cupy as cp
import numpy as np

from src.benchmarks.correctness import correctness_funcs
from src.readers.python.reader import Reader, DEBUG
from src.readers.python.util import type_conv_dict, custom_type_dict, get_kernel_path, get_data_path


class CupyReader:
    def __init__(self, json_path):
        self.spec = self.get_spec(json_path)
        self.kernel_spec = self.spec["kernelSpecification"]

    def get_type_length(self, t):
        return custom_type_dict.get(t, {"length": 1})["length"]

    def get_spec(self, json_path):
        with open(json_path, 'r') as f:
            r = json.load(f)
        return r

    def handle_custom_data_type(self, arg_data, arg):
        t = custom_type_dict[arg["type"]]
        return np.dtype({'names': t["names"], 'formats': t["types"]})

    def handle_vector_data(self, arg):
        # random.seed(10)
        t = arg["fillType"]
        if t == "file":
            with open(get_data_path(self.kernel_spec, arg["path"]), 'r') as f:
                arg_data = f.read().splitlines()
            return self.type_conv_vec(arg_data, arg)
        if t == "random":
            arg_data = [random.random() for _ in range(arg["length"] * self.get_type_length(arg["type"]))]
            return self.type_conv_vec(arg_data, arg)
        if t == "uninitialized":
            arg_data = [0 for _ in range(arg["length"] * self.get_type_length(arg["type"]))]
            return self.type_conv_vec(arg_data, arg)
        if t == "constant":
            arg_data = [eval(str(arg["value"])) for _ in range(arg["length"] * self.get_type_length(arg["type"]))]
            return self.type_conv_vec(arg_data, arg)
        else:
            print("Unsupported vector fill type", arg["fillType"])

    def type_conv_scalar(self, arg_data, arg):
        if arg["type"] in type_conv_dict.keys():
            return type_conv_dict[arg["type"]](arg_data)
        else:  # custom data type
            return (custom_type_dict[arg["type"]]["repr_type"]).view(
                self.handle_custom_data_type(arg_data, arg))(arg_data)

    def populate_args(self):
        return [self.populate_data(arg) for arg in self.kernel_spec["arguments"]]

    def populate_data(self, arg):
        m = arg["memoryType"].lower()
        if m == "vector":
            return self.handle_vector_data(arg)
        if m == "scalar":
            return self.handle_scalar_data(arg)
        else:
            print("Unsupported memory type", arg["memoryType"])

    def generate_compiler_options(self, tuning_config):
        benchmark_config = self.spec["benchmarkConfig"]
        compiler_options = self.kernel_spec["compilerOptions"]
        for (key, val) in tuning_config.items():
            compiler_options.append("-D{}={}".format(key, val))
        for (key, val) in benchmark_config.items():
            compiler_options.append("-D{}={}".format(key, val))
        return compiler_options

    def run(self, tuning_config, testing):
        launch_config = self.get_launch_config(tuning_config)
        return self.run_kernel(launch_config, tuning_config)

    def type_conv_vec(self, arg_data, arg):
        if arg["type"] in type_conv_dict.keys():
            return cp.asarray(arg_data, dtype=type_conv_dict[arg["type"]])
        else:  # custom data type
            return cp.asarray(arg_data, dtype=custom_type_dict[arg["type"]]["repr_type"]).view(
                self.handle_custom_data_type(arg_data, arg))

    def handle_scalar_data(self, arg):
        if arg["type"] == "cudaTextureObject_t":
            raise NotImplementedError
        else:
            return self.type_conv_scalar(arg["value"], arg)

    def execute_kernel(self, args_tuple, launch_config, kernel):
        grid_dim = (launch_config["GRID_SIZE_X"], launch_config["GRID_SIZE_Y"], launch_config["GRID_SIZE_Z"])
        block_dim = (launch_config["BLOCK_SIZE_X"], launch_config["BLOCK_SIZE_Y"], launch_config["BLOCK_SIZE_Z"])
        shared_mem_size = launch_config.get("SHARED_MEMORY_SIZE", 0)
        kernel(grid=grid_dim, block=block_dim, args=args_tuple, shared_mem=shared_mem_size)

    def get_kernel_instance(self, kernel_name, compiler_options):
        jitify = self.spec["benchmarkConfig"].get("jitify", False)
        with open(get_kernel_path(self.spec), 'r') as f:
            module = cp.RawModule(code=f.read(), name_expressions=[],
                                  options=tuple(compiler_options), jitify=jitify)
        func = module.get_function(kernel_name)
        return func

    def launch_kernel(self, args_tuple, launch_config, kernel):
        t0 = time.time()
        for i in range(launch_config.get("ITERATIONS", 10)):
            self.execute_kernel(args_tuple, launch_config, kernel)
        duration = time.time() - t0
        return duration

    def get_kernel(self, compiler_options):
        kernel_name = self.kernel_spec["kernelName"]
        return self.get_kernel_instance(kernel_name, compiler_options)

    def run_kernel(self, launch_config, tuning_config):
        compiler_options = self.generate_compiler_options(tuning_config)
        args = self.populate_args()
        lf_ker = self.get_kernel(compiler_options)
        args_tuple = tuple(args)
        result = self.launch_kernel(args_tuple, launch_config, lf_ker)
        if DEBUG:
            correctness = correctness_funcs[self.kernel_spec["kernelName"]]
            correctness(tuple(args), args_tuple, tuning_config, launch_config)
        return result

    def get_launch_config(self, tuning_config):
        kernel_spec = self.spec["kernelSpecification"]
        for name, value in tuning_config.items():
            locals()[name] = value
        locals()["dataSize"] = self.spec["benchmarkConfig"]["dataSize"]

        launch_config = {
            "GRID_SIZE_X": eval(str(kernel_spec["gridSize"]["X"])),
            "GRID_SIZE_Y": eval(str(kernel_spec["gridSize"]["Y"])),
            "GRID_SIZE_Z": eval(str(kernel_spec["gridSize"]["Z"])),
            "BLOCK_SIZE_X": eval(str(kernel_spec["blockSize"]["X"])),
            "BLOCK_SIZE_Y": eval(str(kernel_spec["blockSize"]["Y"])),
            "BLOCK_SIZE_Z": eval(str(kernel_spec["blockSize"]["Z"])),
        }
        for name, value in launch_config.items():
            locals()[name] = value

        if self.spec["benchmarkConfig"].get("iterations"):
            launch_config["ITERATIONS"] = eval(str(self.spec["benchmarkConfig"]["iterations"]))
        if self.spec["benchmarkConfig"].get("PRECISION"):
            launch_config["PRECISION"] = self.spec["benchmarkConfig"]["PRECISION"]
        if kernel_spec.get("sharedMemory"):
            launch_config["SHARED_MEMORY_SIZE"] = eval(str(kernel_spec["sharedMemory"]))

        return launch_config
