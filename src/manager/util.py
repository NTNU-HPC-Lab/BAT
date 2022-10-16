import json

import numpy as np


def get_benchmark_path(spec):
    return "benchmarks/{}".format(spec["General"]["BenchmarkName"])


def get_kernel_path(spec):
    return "{}/{}".format(get_benchmark_path(spec), spec["KernelSpecification"]["KernelFile"])


def get_data_path(spec, local_data_path):
    return "{}/{}".format(get_benchmark_path(spec), local_data_path)


def print_json(j):
    print(json.dumps(j, indent=4, sort_keys=True))

def get_spec(json_path):
    with open(json_path, 'r') as f:
        r = json.load(f)
    return r

def get_search_spec():
    with open('search-settings.json', 'r') as f:
        return json.loads(f.read())

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
    "float": np.float,
    "double": np.float64,
    "quad": np.longdouble,
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
