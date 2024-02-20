import logging
from enum import Enum
import numpy as np
try:
    import cupy as cp
except:
    print("No cuda found")

from batbench.config_space.arguments import Arguments

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
        self.args = Arguments(spec["KernelSpecification"]["KernelName"])
        self.backend = spec["KernelSpecification"]["Language"]
        self.args.set_backend(self.backend)
        self.spec_args = spec["KernelSpecification"]["Arguments"]
        self.cmem_args = {}

    def handle_custom_data_type(self, arg_data, arg):
        arg_type = custom_type_dict[arg["Type"]]
        return np.dtype({
            'names': arg_type["names"],
            'formats': arg_type["types"],
            'aligned': True
        })

    def handle_vector_data(self, arg):
        # random.seed(10)
        if arg["FillType"] not in FillType._value2member_map_:
            logger.error("Unsupported vector fill type %s", arg["FillType"])
            raise ValueError(f"Unsupported fill type: {arg['FillType']}")

        fill = FillType(arg["FillType"])
        size_length = arg["Size"] * self.get_type_length(arg["Type"])
        arg_data = []

        if fill == FillType.BINARY_RAW:
            raise NotImplementedError
            #with open(get_data_path(self.spec, arg["DataSource"]), 'r') as f:
            #    arg_data = f.read().splitlines()
        if fill == FillType.RANDOM:
            arg_data = np.random.randn(size_length)
        if fill == FillType.CONSTANT:
            fill_value = eval(str(arg["FillValue"]))
            arg_data = np.zeros(size_length) if fill_value == 0 else np.full(
                size_length, fill_value)
        if fill == FillType.GENERATOR:
            f_vec = np.vectorize(lambda i: eval(str(arg["DataSource"]), {"i": i}))
            arr = np.arange(0, size_length)
            return f_vec(arr)

        arr = []
        if self.backend == "CUDA":
            arr = cp.asarray(self.type_conv_vec(arg_data, arg))
        else:
            arr = np.asarray(self.type_conv_vec(arg_data, arg))
        return arr

    def type_conv_scalar(self, arg_data, arg):
        if arg["Type"] in type_conv_dict:
            return type_conv_dict[arg["Type"]](arg_data)
        # custom data type
        return np.array(arg_data, custom_type_dict[arg["Type"]]["repr_type"]).view(
            self.handle_custom_data_type(arg_data, arg))

    def populate_args(self) -> Arguments:
        if self.args.empty():
            for _, arg in enumerate(self.spec_args):
                name = arg["Name"]
                print(f"Populating arg {name}", end="\r")
                pop_arg = self.populate_data(arg)
                self.args.add(key=name, value=pop_arg,
                              cmem=arg.get("MemType", "") == "Constant",
                              output=arg.get("Output",0) == 1)
        return self.args


    def populate_data(self, arg):
        mem_type = arg["MemoryType"]
        if mem_type == "Vector":
            return self.handle_vector_data(arg)
        if mem_type == "Scalar":
            return self.handle_scalar_data(arg)
        raise UnsupportedMemoryTypeError(f"Unsupported memory type {arg['MemoryType']}")

    def type_conv_vec(self, arg_data, arg):
        if arg["Type"] in type_conv_dict:
            return arg_data.astype(type_conv_dict[arg["Type"]])
        # custom data type
        return arg_data.astype(custom_type_dict[arg["Type"]]["repr_type"]).view(
            self.handle_custom_data_type(arg_data, arg))

    def handle_scalar_data(self, arg):
        if arg["Type"] == "cudaTextureObject_t":
            raise NotImplementedError
        return self.type_conv_scalar(arg["FillValue"], arg)

    def get_type_length(self, custom_type):
        return custom_type_dict.get(custom_type, { "TypeSize": 1 })["TypeSize"]
