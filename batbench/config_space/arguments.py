import os
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cupy as cp

class Arguments():
    def __init__(self, kernel_name: str, args: Optional[Dict[str, Dict]] = None) -> None:
        if args is None:
            args = {}
        self.args = args
        self.reference_path = f"./references/{kernel_name}"
        os.makedirs(self.reference_path, exist_ok=True)

        self.initial_values = {} # The value of the output variables before the kernel is run
        self.output_args = set() # The names of the output variables
        self.cmem_args = set() # The names of the constant memory variables
        # The value of the output variables after the reference kernel is run
        self.reference_values = {}

        for key, arg in self.args.items():
            self.add(key, arg["value"], arg["cmem"], arg["output"], arg["index"])
        self.index = 0
        self.backend = "Cupy"

    def empty(self) -> bool:
        return not bool(self.args)

    def set_backend(self, backend_type):
        self.backend = backend_type

    def add_reference_value(self, key: str, value: List) -> None:
        assert key in self.output_args
        self.reference_values[key] = np.copy(value)

    def add(self, key: str, value: Union[np.ndarray, np.generic], 
            cmem=False, output=False, index: int = -1) -> None:
        if output:
            self.output_args.add(key)
            self.initial_values[key] = np.copy(value)
        if cmem:
            self.cmem_args.add(key)
        if index == -1:
            index = self.index
            self.index += 1
        self.args[key] = {
            "index": index,
            "key": key,
            "value": value,
            "cmem": cmem,
            "output": output
        }

    def get_constant_values_list(self) -> List:
        return [self.args[key]["value"] for key in self.cmem_args]

    def get_constant_names_list(self) -> List:
        return list(self.cmem_args)

    def get_constant_dict(self) -> Dict:
        temp_d = {}
        for key in self.cmem_args:
            val = self.args[key]["value"]
            if self.backend == "CUDA":
                val: np.ndarray = cp.asnumpy(val)
            else:
                val:np.ndarray = np.asarray(val)
            print(f"val type: {type(val)}")
            assert isinstance(val, np.ndarray)
            temp_d[key] = val
        return temp_d

    def get_list(self) -> List:
        return [arg["value"] for arg in self.args.values()]

    def get_function_args(self) -> Tuple[List, List]:
        return self.get_list(), self.get_constant_dict()

    def get(self, key: str = "") -> Dict:
        if key in self.args:
            return self.args[key]
        return self.args

    def check_output(self, post_run_values: List) -> bool:
        # There should be at least one output variable
        if len(self.output_args) == 0:
            error_message = 'No output variables found.'
            logging.error(error_message)
            raise ValueError(error_message)

        for key in self.output_args:
            # The output variable should be in the args
            if key not in self.args or not self.args[key]["output"]:
                error_message = f'Invalid output variable: {key}.'
                logging.error(error_message)
                raise ValueError(error_message)

            # The output variable should have changed
            if np.array_equal(post_run_values[self.args[key]["index"]],
                              self.initial_values[key]):
                error_message = f'Output variable {key} did not change.'
                logging.error(error_message)
                raise ValueError(error_message)

            # The output variable should be the same as the reference kernel
            if self.reference_values and key not in self.reference_values:
                error_message = f'Reference value for {key} not found.'
                logging.error(error_message)
                raise ValueError(error_message)

            if self.reference_values and not np.array_equal(
                post_run_values[self.args[key]["index"]], self.reference_values[key]):
                error_message = f'Output variable {key} does not match reference value.'
                logging.error(error_message)
                raise ValueError(error_message)

        return True
