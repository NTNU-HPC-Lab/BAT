import json


def get_benchmark_path(kernel_spec):
    return "./src/benchmarks/{}".format(kernel_spec["benchmarkName"])


def get_kernel_path(kernel_spec):
    return "{}/{}".format(get_benchmark_path(kernel_spec), kernel_spec["kernelFile"])


def get_data_path(kernel_spec, local_data_path):
    return "{}/{}".format(get_benchmark_path(kernel_spec), local_data_path)


def print_json(j):
    print(json.dumps(j, indent=4, sort_keys=True))