import json
import random

from abc import ABC, abstractmethod

import numpy as np

from src.readers.python.util import custom_type_dict, get_data_path, type_conv_dict

DEBUG = False



class Reader(ABC):
    _kernel_cache = None

    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.spec = json.load(f)
        self.kernel_spec = self.spec["kernelSpecification"]

    @abstractmethod
    def launch_kernel(self, args_tuple, launch_config, kernel):
        pass

    @abstractmethod
    def execute_kernel(self, args_tuple, launch_config, kernel):
        pass

    @abstractmethod
    def get_kernel_instance(self, kernel_name, compiler_options):
        pass

    @abstractmethod
    def get_launch_config(self, tuning_config):
        pass

    @abstractmethod
    def get_kernel(self, compiler_options):
        pass

    @abstractmethod
    def run_kernel(self, launch_config, tuning_config):
        pass

    @staticmethod
    def get_type_length(t):
        return custom_type_dict.get(t, {"length": 1})["length"]

    @staticmethod
    def get_spec(json_path):
        with open(json_path, 'r') as f:
            r = json.load(f)
        return r

    @staticmethod
    def handle_custom_data_type(arg_data, arg):
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

    def handle_scalar_data(self, arg):
        return self.type_conv_scalar(arg["value"], arg)

    def type_conv_vec(self, arg_data, arg):
        if arg["type"] in type_conv_dict.keys():
            return np.asarray(arg_data, dtype=type_conv_dict[arg["type"]])
        else:  # custom data type
            return np.asarray(arg_data, dtype=custom_type_dict[arg["type"]]["repr_type"]).view(
                self.handle_custom_data_type(arg_data, arg))

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
