import numpy as np
import cupy as cp
import random

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
            types = [cp.float32, cp.float32, cp.float32]
            return custom_data_type(arg_data, names, types, cp.float32)
        case "float4":
            names = ['x', 'y', 'z', 'w']
            types = [cp.float32, cp.float32, cp.float32, cp.float32]
            return custom_data_type(arg_data, names, types, cp.float32)


def type_conv(arg_data, arg):
    if arg["type"] in type_conv_dict.keys():
        return cp.asarray(arg_data, dtype=type_conv_dict[arg["type"]])
    else:  # custom data type
        return handle_custom_data_type(arg_data, arg)


def get_type_length(t):
    type_length = 1
    if t == "float4":
        type_length = 4
    if t == "float3":
        type_length = 3
    return type_length


def handle_vector_data(arg):
    match arg["fillType"]:
        case "file":
            with open(arg["path"], 'r') as f:
                arg_data = f.read().splitlines()
                print(len(arg_data), len(arg_data) // 128)
            return type_conv(arg_data, arg)
        case "random":
            arg_data = [random.random() for _ in range(arg["length"] * get_type_length(arg["type"]))]
            print(len(arg_data), len(arg_data) // get_type_length(arg["type"]), arg["type"])
            return type_conv(arg_data, arg)
        case "uninitialized":
            arg_data = [0 for i in range(arg["length"] * get_type_length(arg["type"]))]
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
    data_size = kernel_spec["dataSize"]
    block_size = tuning_config["BLOCK_SIZE"]
    launch_config = {
        "GRID_SIZE_X": (data_size + block_size - 1) // block_size,
        "GRID_SIZE_Y": kernel_spec["gridSize"]["Y"],
        "GRID_SIZE_Z": kernel_spec["gridSize"]["Z"],
        "BLOCK_SIZE_X": block_size,
        "BLOCK_SIZE_Y": kernel_spec["blockSize"]["Y"],
        "BLOCK_SIZE_Z": kernel_spec["blockSize"]["Z"]
    }
    return launch_config
