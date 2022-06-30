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
    "float": np.single,
    "double": np.double,
    "longdouble": np.longdouble,
    "csingle": np.csingle,
    "cdouble": np.cdouble,
    "clongdouble": np.clongdouble
}


def populate_args(kernel_spec):
    return [populate_data(arg) for arg in kernel_spec["arguments"]]


def custom_data_type(arg_data, names, types, repr_type):
    custom = cp.dtype({'names': names, 'formats': types})
    return cp.asarray(arg_data, dtype=repr_type).view(custom)


def handle_custom_data_type(arg_data, arg):
    match arg["type"]:
        case "float3":
            names = ['x', 'y', 'z']
            types = [np.float32] * 3
            return custom_data_type(arg_data, names, types, np.float32)
        case "float4":
            names = ['x', 'y', 'z', 'w']
            types = [np.float32] * 4
            return custom_data_type(arg_data, names, types, np.float32)
        case "double3":
            names = ['x', 'y', 'z']
            types = [np.float64] * 3
            return custom_data_type(arg_data, names, types, np.float64)
        case "double4":
            names = ['x', 'y', 'z', 'w']
            types = [np.float64] * 4
            return custom_data_type(arg_data, names, types, np.float64)


def type_conv(arg_data, arg):
    if arg["type"] in type_conv_dict.keys():
        return cp.asarray(arg_data, dtype=type_conv_dict[arg["type"]])
    else:  # custom data type
        return handle_custom_data_type(arg_data, arg)


def get_type_length(t):
    type_length = 1
    match t:
        case "float4":
            type_length = 4
        case "float3":
            type_length = 3
        case "double4":
            type_length = 4
        case "double3":
            type_length = 3
    return type_length


def handle_vector_data(arg):
    match arg["fillType"]:
        case "file":
            with open(arg["path"], 'r') as f:
                arg_data = f.read().splitlines()
                if DEBUG:
                    print(len(arg_data), len(arg_data) // 128)
            return type_conv(arg_data, arg)
        case "random":
            arg_data = [random.random() for _ in range(arg["length"] * get_type_length(arg["type"]))]
            if DEBUG:
                print(len(arg_data), len(arg_data) // get_type_length(arg["type"]), arg["type"])
            return type_conv(arg_data, arg)
        case "uninitialized":
            arg_data = [0 for _ in range(arg["length"] * get_type_length(arg["type"]))]
            if DEBUG:
                print(len(arg_data), len(arg_data) // get_type_length(arg["type"]), arg["type"])
            return type_conv(arg_data, arg)
        case _:
            print("Unsupported vector memory type", arg["memoryType"])


def handle_scalar_data(arg):
    if arg["type"] == "cudaTextureObject_t":
        raise NotImplementedError
    else:
        arg_data = arg["value"]
        return type_conv(arg_data, arg)


def populate_data(arg):
    match arg["memoryType"]:
        case "Vector":
            return handle_vector_data(arg)
        case "Scalar":
            return handle_scalar_data(arg)
        case _:
            print("Unsupported memory type", arg["memoryType"])


def get_launch_config(kernel_spec, tuning_config):
    global_vars = tuning_config
    for name, value in tuning_config.items():
        globals()[name] = value
    globals()["dataSize"] = kernel_spec["dataSize"]

    print(global_vars)
    launch_config = {
        "GRID_SIZE_X": eval(str(kernel_spec["gridSize"]["X"])),
        "GRID_SIZE_Y": eval(str(kernel_spec["gridSize"]["Y"])),
        "GRID_SIZE_Z": eval(str(kernel_spec["gridSize"]["Z"])),
        "BLOCK_SIZE_X": eval(str(kernel_spec["blockSize"]["X"])),
        "BLOCK_SIZE_Y": eval(str(kernel_spec["blockSize"]["Y"])),
        "BLOCK_SIZE_Z": eval(str(kernel_spec["blockSize"]["Z"])),
    }
    print(launch_config)
    return launch_config


def get_search_space(json_path):
    with open(json_path, 'r') as f:
        r = json.load(f)
    return r["configurationSpace"]


def get_kernel_spec(json_path):
    with open(json_path, 'r') as f:
        r = json.load(f)
    return r["kernelSpecification"]


def get_spec(json_path):
    with open(json_path, 'r') as f:
        r = json.load(f)
    return r