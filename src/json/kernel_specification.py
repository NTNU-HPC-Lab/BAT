import numpy as np
import cupy as cp
import json
import random

DEBUG = 0


def get_spec(json_path):
    with open(json_path, 'r') as f:
        r = json.load(f)
    return r


def get_launch_config(kernel_spec, tuning_config):
    for name, value in tuning_config.items():
        locals()[name] = value
    locals()["dataSize"] = kernel_spec["dataSize"]

    launch_config = {
        "GRID_SIZE_X": eval(str(kernel_spec["gridSize"]["X"])),
        "GRID_SIZE_Y": eval(str(kernel_spec["gridSize"]["Y"])),
        "GRID_SIZE_Z": eval(str(kernel_spec["gridSize"]["Z"])),
        "BLOCK_SIZE_X": eval(str(kernel_spec["blockSize"]["X"])),
        "BLOCK_SIZE_Y": eval(str(kernel_spec["blockSize"]["Y"])),
        "BLOCK_SIZE_Z": eval(str(kernel_spec["blockSize"]["Z"])),
    }
    return launch_config


class Handler:
    type_conv_dict = {
        "bool": np.bool_,
        "byte": np.byte,
        "short": np.short,
        "ushort": np.ushort,
        "int": np.intc,
        "uint": np.uintc,
        "long": np.int_,
        "ulong": np.uint,
        "longlong": np.longlong,
        "ulonglong": np.ulonglong,
        "half": np.half,
        "float": np.single,
        "double": np.double,
        "longdouble": np.longdouble,
        "csingle": np.csingle,
        "cdouble": np.cdouble,
        "clongdouble": np.clongdouble
    }
    type_conv_dict_custom = {}
    type_conv_dict_custon_repr_type = {}
    type_conv_dict_custom_length = {}

    def __init__(self, spec):
        self.spec = spec
        for t in spec["metadata"]["types"]:
            names = []
            field_types = []
            for name, field_type in t["fields"].items():
                names.append(name)
                field_types.append(self.type_conv_dict[field_type])

            self.type_conv_dict_custom[t["name"]] = cp.dtype({'names': names, 'formats': field_types})
            self.type_conv_dict_custon_repr_type[t["name"]] = self.type_conv_dict[t["repr_type"]]
            self.type_conv_dict_custom_length[t["name"]] = len(t["fields"].items())

    def get_type_length(self, t):
        return self.type_conv_dict_custom_length.get(t, 1)

    def populate_args(self, kernel_spec):
        return [self.populate_data(arg) for arg in kernel_spec["arguments"]]

    def type_conv(self, arg_data, arg):
        if arg["type"] in self.type_conv_dict.keys():
            return cp.asarray(arg_data, dtype=self.type_conv_dict[arg["type"]])
        else:  # custom data type
            return cp.asarray(arg_data, dtype=self
                              .type_conv_dict_custon_repr_type[arg["type"]]) \
                .view(self.type_conv_dict_custom[arg["type"]])

    def handle_vector_data(self, arg):
        match arg["fillType"]:
            case "file":
                with open(arg["path"], 'r') as f:
                    arg_data = f.read().splitlines()
                return self.type_conv(arg_data, arg)
            case "random":
                arg_data = [random.random() for _ in range(arg["length"] * self.get_type_length(arg["type"]))]
                return self.type_conv(arg_data, arg)
            case "uninitialized":
                arg_data = [0 for _ in range(arg["length"] * self.get_type_length(arg["type"]))]
                return self.type_conv(arg_data, arg)
            case _:
                print("Unsupported vector memory type", arg["memoryType"])

    def handle_scalar_data(self, arg):
        match arg["type"]:
            case "cudaTextureObject_t":
                raise NotImplementedError
            case _:
                return self.type_conv(arg["value"], arg)

    def populate_data(self, arg):
        match arg["memoryType"]:
            case "Vector":
                return self.handle_vector_data(arg)
            case "Scalar":
                return self.handle_scalar_data(arg)
            case _:
                print("Unsupported memory type", arg["memoryType"])
