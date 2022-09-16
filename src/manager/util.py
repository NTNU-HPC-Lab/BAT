import json

import numpy as np


def get_benchmark_path(spec):
    return "benchmarks/{}".format(spec["general"]["benchmarkName"])


def get_kernel_path(spec):
    return "{}/{}".format(get_benchmark_path(spec), spec["kernelSpecification"]["kernelFile"])


def get_data_path(spec, local_data_path):
    return "{}/{}".format(get_benchmark_path(spec), local_data_path)


def print_json(j):
    print(json.dumps(j, indent=4, sort_keys=True))


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
