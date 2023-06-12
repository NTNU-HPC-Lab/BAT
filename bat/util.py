import json

import numpy as np

benchmark_root_path = "bat/benchmarks"
schema_path = "bat/schemas/TuningSchema"

def get_benchmark_path(spec):
    return f"{benchmark_root_path}/{spec['General']['BenchmarkName']}"

def get_kernel_path(spec):
    return "{}/{}".format(get_benchmark_path(spec), spec["KernelSpecification"]["KernelFile"])

def get_data_path(spec, local_data_path):
    return "{}/{}".format(get_benchmark_path(spec), local_data_path)

def get_spec_by_name(name):
    return get_spec(f"{benchmark_root_path}/{name}/{name}-CAFF.json")

def print_json(j):
    print(json.dumps(j, indent=4, sort_keys=True))

def get_spec(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def write_spec(spec, path):
    with open(path, 'w') as f:
        f.write(json.dumps(spec, indent=4, sort_keys=True))

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
        "length": 2,
        "repr_type": np.float32
    },

    "float3": {
        "names": ['x', 'y', 'z'],
        "types": [np.float32] * 3,
        "length": 3,
        "repr_type": np.float32
    },

    "float4": {
        "names": ['x', 'y', 'z', 'w'],
        "types": [np.float32] * 4,
        "length": 4,
        "repr_type": np.float32
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
