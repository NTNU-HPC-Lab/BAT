import numpy as np
import cupy as cp
import random

from src.manager import get_data_path


type_conv_dict = {
    "bool": np.bool_,
    "byte": np.byte,
    "int16": np.short,
    "uint16": np.ushort,
    "int32": np.intc,
    "uint32": np.uintc,
    "int64": np.int_,
    "uint64": np.uint,
    "int128": np.longlong,
    "uint128": np.ulonglong,
    "half": np.half,
    "float": np.float32,
    "double": np.float64,
    "quad": np.longdouble,
    "csingle": np.csingle,
    "cdouble": np.cdouble,
    "clongdouble": np.clongdouble
}

custom_type_dict = {
    "float2": {
        "names": ['x', 'y'],
        "types": [np.float32] * 2,
        "TypeSize": 2,
        "repr_type": np.float32
    },

    "float3": {
        "names": ['x', 'y', 'z'],
        "types": [np.float32] * 3,
        "TypeSize": 3,
        "repr_type": np.float32
    },

    "float4": {
        "names": ['x', 'y', 'z', 'w'],
        "types": [np.float32] * 4,
        "TypeSize": 4,
        "repr_type": np.float32
    },

    "double2": {
        "names": ['x', 'y'],
        "types": [np.float64] * 2,
        "TypeSize": 2,
        "repr_type": np.float64
    },

    "double3": {
        "names": ['x', 'y', 'z'],
        "types": [np.float64] * 3,
        "TypeSize": 3,
        "repr_type": np.float64
    },

    "double4": {
        "names": ['x', 'y', 'z', 'w'],
        "types": [np.float64] * 4,
        "TypeSize": 4,
        "repr_type": np.float64
    },
}
class ArgHandler:

    def __init__(self, spec):
        self.spec = spec
        self.args = []

    def handle_custom_data_type(self, arg_data, arg):
        t = custom_type_dict[arg["Type"]]
        return np.dtype({
            'names': t["names"],
            'formats': t["types"]
        })

    def handle_vector_data(self, arg):
        # random.seed(10)
        t = arg["FillType"]
        arg_data = []
        if t not in ("BinaryRaw", "Random", "Uninitialized", "Constant", "Generator"):
            print("Unsupported vector fill type", arg["fillType"])
            return
        if t == "BinaryRaw":
            with open(get_data_path(self.spec, arg["DataSource"]), 'r') as f:
                arg_data = f.read().splitlines()
        if t == "Random":
            arg_data = [random.random() for _ in range(arg["Size"] * self.get_type_length(arg["Type"]))]
        if t == "Uninitialized":
            arg_data = [0 for _ in range(arg["Size"] * self.get_type_length(arg["Type"]))]
        if t == "Constant":
            arg_data = [eval(str(arg["FillValue"])) for _ in range(arg["Size"] * self.get_type_length(arg["Type"]))]
        if t == "Generator":
            arg_data = [eval(str(arg["DataSource"]), {"i": i}) for i in range(arg["Size"] * self.get_type_length(arg["Type"]))]
        return self.type_conv_vec(arg_data, arg)

    def type_conv_scalar(self, arg_data, arg):
        if arg["Type"] in type_conv_dict.keys():
            return type_conv_dict[arg["Type"]](arg_data)
        else:    # custom data type
            return (custom_type_dict[arg["Type"]]["repr_type"]).view(self.handle_custom_data_type(arg_data, arg))(arg_data)

    def populate_args(self, args):
        if self.args == []:
            self.args = [self.populate_data(arg) for arg in args]
        return self.args

    def populate_data(self, arg):
        m = arg["MemoryType"]
        if m == "Vector":
            return self.handle_vector_data(arg)
        if m == "Scalar":
            return self.handle_scalar_data(arg)
        else:
            print("Unsupported memory type", arg["MemoryType"])

    def type_conv_vec(self, arg_data, arg):
        if arg["Type"] in type_conv_dict.keys():
            return cp.asarray(arg_data, dtype=type_conv_dict[arg["Type"]])
        else:    # custom data type
            return cp.asarray(arg_data, dtype=custom_type_dict[arg["Type"]]["repr_type"]).view(self.handle_custom_data_type(arg_data, arg))

    def handle_scalar_data(self, arg):
        if arg["Type"] == "cudaTextureObject_t":
            raise NotImplementedError
        else:
            return self.type_conv_scalar(arg["FillValue"], arg)

    def get_type_length(self, t):
        return custom_type_dict.get(t, { "TypeSize": 1 })["TypeSize"]

