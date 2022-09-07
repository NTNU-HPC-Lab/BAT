import json

from src.readers.python.util import get_kernel_path, get_data_path

def get_spec(json_path):
    with open(json_path, 'r') as f:
        r = json.load(f)
    return r

