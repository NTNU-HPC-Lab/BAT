import numpy as np
import cupy as cp
import json
import random

DEBUG = 0

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
    "float": np.float,
    "double": np.float64,
    "longdouble": np.longdouble,
    "csingle": np.csingle,
    "cdouble": np.cdouble,
    "clongdouble": np.clongdouble
}

custom_type_dict = {
    "float3": {
        "names": ['x', 'y', 'z'],
        "types": [np.float] * 3,
        "length": 3,
        "repr_type": np.float
    },

    "float4": {
        "names": ['x', 'y', 'z', 'w'],
        "types": [np.float] * 4,
        "length": 4,
        "repr_type": np.float
    },

    "double3": {
        "names": ['x', 'y', 'z'],
        "types": [np.float64] * 3,
        "length": 3,
        "repr_type": np.float64
    },

    "double4": {
        "names": ['x', 'y', 'z', 'w'],
        "types": [np.float64] * 4,
        "length": 4,
        "repr_type": np.float64
    },
}


def populate_args(kernel_spec):
    return [populate_data(arg) for arg in kernel_spec["arguments"]]


def handle_custom_data_type(arg_data, arg):
    t = custom_type_dict[arg["type"]]
    custom = np.dtype({'names': t["names"], 'formats': t["types"]})
    return cp.asarray(arg_data, dtype=t["repr_type"]).view(custom)


def type_conv(arg_data, arg):
    if arg["type"] in type_conv_dict.keys():
        return cp.asarray(arg_data, dtype=type_conv_dict[arg["type"]])
    else:  # custom data type
        return handle_custom_data_type(arg_data, arg)


def get_type_length(t):
    return custom_type_dict.get(t, {"length": 1})["length"]


def handle_vector_data(arg):
    random.seed(10)
    match arg["fillType"]:
        case "file":
            with open(arg["path"], 'r') as f:
                arg_data = f.read().splitlines()
            return type_conv(arg_data, arg)
        case "random":
            arg_data = [random.random() for _ in range(arg["length"] * get_type_length(arg["type"]))]
            return type_conv(arg_data, arg)
        case "uninitialized":
            arg_data = [0 for _ in range(arg["length"] * get_type_length(arg["type"]))]
            return type_conv(arg_data, arg)
        case "constant":
            arg_data = [eval(str(arg["value"])) for _ in range(arg["length"] * get_type_length(arg["type"]))]
            return type_conv(arg_data, arg)
        case _:
            print("Unsupported vector fill type", arg["fillType"])


def handle_scalar_data(arg):
    if arg["type"] == "cudaTextureObject_t":
        raise NotImplementedError
    else:
        return type_conv(arg["value"], arg)


def populate_data(arg):
    match arg["memoryType"].lower():
        case "vector":
            return handle_vector_data(arg)
        case "scalar":
            return handle_scalar_data(arg)
        case _:
            print("Unsupported memory type", arg["memoryType"])


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


def get_spec(json_path):
    with open(json_path, 'r') as f:
        r = json.load(f)
    return r
