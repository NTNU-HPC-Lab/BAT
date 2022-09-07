import numpy as np
import cupy as cp
import random

from src.manager import get_data_path


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
    "float2": {
        "names": ['x', 'y'],
        "types": [np.float] * 2,
        "length": 2,
        "repr_type": np.float
    },

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

    "double2": {
        "names": ['x', 'y'],
        "types": [np.float64] * 2,
        "length": 2,
        "repr_type": np.float64
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
class ArgHandler:

    def __init__(self, spec):
        self.spec = spec

    def handle_custom_data_type(self, arg_data, arg):
        t = custom_type_dict[arg["type"]]
        return np.dtype({
            'names': t["names"],
            'formats': t["types"]
        })

    def handle_vector_data(self, arg):
        # random.seed(10)
        t = arg["fillType"]
        arg_data = []
        if t not in ("file", "random", "uninitialized", "constant"):
            print("Unsupported vector fill type", arg["fillType"])
            return
        if t == "file":
            with open(get_data_path(self.spec, arg["path"]), 'r') as f:
                arg_data = f.read().splitlines()
        if t == "random":
            arg_data = [random.random() for _ in range(arg["length"] * self.get_type_length(arg["type"]))]
        if t == "uninitialized":
            arg_data = [0 for _ in range(arg["length"] * self.get_type_length(arg["type"]))]
        if t == "constant":
            arg_data = [eval(str(arg["value"])) for _ in range(arg["length"] * self.get_type_length(arg["type"]))]
        return self.type_conv_vec(arg_data, arg)

    def type_conv_scalar(self, arg_data, arg):
        if arg["type"] in type_conv_dict.keys():
            return type_conv_dict[arg["type"]](arg_data)
        else:    # custom data type
            return (custom_type_dict[arg["type"]]["repr_type"]).view(self.handle_custom_data_type(arg_data, arg))(arg_data)

    def populate_args(self, args):
        return [self.populate_data(arg) for arg in args]

    def populate_data(self, arg):
        m = arg["memoryType"].lower()
        if m == "vector":
            return self.handle_vector_data(arg)
        if m == "scalar":
            return self.handle_scalar_data(arg)
        else:
            print("Unsupported memory type", arg["memoryType"])

    def type_conv_vec(self, arg_data, arg):
        if arg["type"] in type_conv_dict.keys():
            return cp.asarray(arg_data, dtype=type_conv_dict[arg["type"]])
        else:    # custom data type
            return cp.asarray(arg_data, dtype=custom_type_dict[arg["type"]]["repr_type"]).view(self.handle_custom_data_type(arg_data, arg))

    def handle_scalar_data(self, arg):
        if arg["type"] == "cudaTextureObject_t":
            raise NotImplementedError
        else:
            return self.type_conv_scalar(arg["value"], arg)

    def get_type_length(self, t):
        return custom_type_dict.get(t, { "length": 1 })["length"]

