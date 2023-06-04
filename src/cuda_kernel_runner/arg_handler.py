import numpy as np
import cupy as cp
from enum import Enum
from typing import List, Dict, Tuple
import logging

# Define the logger
logger = logging.getLogger(__name__)

# Define Enum for FillType

type_conv_dict = {
    "bool": np.bool_,
    "byte": np.byte,
    "int8": np.byte,
    "uint8": np.ubyte,
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

class FillType(Enum):
    BINARY_RAW = "BinaryRaw"
    RANDOM = "Random"
    UNINITIALIZED = "Uninitialized"
    CONSTANT = "Constant"
    GENERATOR = "Generator"


class UnsupportedMemoryTypeError(Exception):
    pass

class ArgHandler:

    def __init__(self, spec):
        self.spec = spec
        self.args = []
        self.spec_args = spec["KernelSpecification"]["Arguments"]
        self.cmem_args = {}

    def handle_custom_data_type(self, arg_data, arg):
        t = custom_type_dict[arg["Type"]]
        return np.dtype({
            'names': t["names"],
            'formats': t["types"]
        })

    def handle_vector_data(self, arg):
        # random.seed(10)
        try:
            t = FillType(arg["FillType"])
        except ValueError as e:
            logger.error("Unsupported vector fill type %s", arg["fillType"])
            return
        
        size_length = arg["Size"] * self.get_type_length(arg["Type"])
        arg_data = []


        if t == FillType.BINARY_RAW:
            raise NotImplementedError
            #with open(get_data_path(self.spec, arg["DataSource"]), 'r') as f:
            #    arg_data = f.read().splitlines()
        if t == FillType.RANDOM:
            arg_data = np.random.randn(size_length)
        if t == FillType.CONSTANT:
            c = eval(str(arg["FillValue"]))
            arg_data = np.zeros(size_length) if c == 0 else np.full(size_length, c)
        if t == FillType.GENERATOR:
            f_vec = np.vectorize(lambda i: eval(str(arg["DataSource"]), {"i": i}))
            arr = np.arange(0, size_length)
            return f_vec(arr)
        
        return cp.asarray(self.type_conv_vec(arg_data, arg))

    def type_conv_scalar(self, arg_data, arg):
        if arg["Type"] in type_conv_dict.keys():
            return type_conv_dict[arg["Type"]](arg_data)
        else:    # custom data type
            return (custom_type_dict[arg["Type"]]["repr_type"]).view(self.handle_custom_data_type(arg_data, arg))(arg_data)

    def populate_args(self) -> Tuple[List, Dict]:
        if self.args == [] and self.cmem_args == {}:
            pop_args = []
            pop_cmem_args = {}
            for i, arg in enumerate(self.spec_args):
                name = arg["Name"]
                print(f"Populating arg {name}", end="\r")
                pop_arg = self.populate_data(arg)
                if arg.get("MemType", "") == "Constant":
                    pop_cmem_args[arg["Name"]] = pop_arg
                pop_args.append(pop_arg)
            self.args = pop_args
            self.cmem_args = pop_cmem_args
        return self.args, self.cmem_args

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
            return arg_data.astype(type_conv_dict[arg["Type"]])
        else:    # custom data type
            return arg_data.astype(custom_type_dict[arg["Type"]]["repr_type"]).view(self.handle_custom_data_type(arg_data, arg))

    def handle_scalar_data(self, arg):
        if arg["Type"] == "cudaTextureObject_t":
            raise NotImplementedError
        else:
            return self.type_conv_scalar(arg["FillValue"], arg)

    def get_type_length(self, t):
        return custom_type_dict.get(t, { "TypeSize": 1 })["TypeSize"]

