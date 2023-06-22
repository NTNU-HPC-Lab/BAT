import json

BENCHMARK_ROOT_PATH = "batbench/benchmarks"
SCHEMA_PATH = "batbench/schemas/TuningSchema"

def get_benchmark_path(spec):
    return f"{BENCHMARK_ROOT_PATH}/{spec['General']['BenchmarkName']}"

def get_kernel_path(spec):
    return f'{get_benchmark_path(spec)}/{spec["KernelSpecification"]["KernelFile"]}'

def get_data_path(spec, local_data_path):
    return f"{get_benchmark_path(spec)}/{local_data_path}"

def get_spec_by_name(name):
    return get_spec(f"{BENCHMARK_ROOT_PATH}/{name}/{name}-CAFF.json")

def print_json(json_dict):
    print(json.dumps(json_dict, indent=4, sort_keys=True))

def get_spec(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_spec(spec, path):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(spec, indent=4, sort_keys=True))
