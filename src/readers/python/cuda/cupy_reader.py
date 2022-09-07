import json
import random
import time
import itertools
from collections import namedtuple

import cupy as cp
import numpy as np

from src.benchmarks.correctness import correctness_funcs
from src.readers.python.util import get_kernel_path, get_data_path

def get_spec(json_path):
    with open(json_path, 'r') as f:
        r = json.load(f)
    return r

def core(json_path, tuning_config, testing):
    return CudaKernel(json_path).run(tuning_config, testing)

