from builtins import str
import ast
import json
import pandas as pd
import copy
from collections import OrderedDict

import kernel_tuner
from kernel_tuner.interface import Options, _kernel_options, _device_options, _tuning_options
from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner import util

from src.manager import get_kernel_path
from src.manager import Manager
from src.result import Result


class KernelBackend:
    def __init__(self, args, manager):
        self.prog_args = args
        self.manager = manager
        self.spec = manager.spec
        self.kernel_spec = self.spec["KernelSpecification"]
        self.original_compiler_options = copy.deepcopy(self.kernel_spec["CompilerOptions"])
        kernel_spec = self.kernel_spec
        if kernel_spec["Language"] != "CUDA":
            raise NotImplementedError(
                "Currently only CUDA kernels have been implemented")
        # add tune params
        tune_params = OrderedDict(self.manager.config_space.get_parameters())
        # get arguments
        iterations = eval(
            str(self.manager.spec["BenchmarkConfig"]["iterations"])
        )  # number of times each kernel configuration is ran
        block_size_names = list(n for n in kernel_spec["LocalSize"].values() if not n.isdigit())

        observers = None
        #while len(block_size_names) < 3:
            #    block_size_names.append("")

        # check for forbidden names in tune parameters
        util.check_tune_params_list(tune_params, observers)

        # check whether block_size_names are used as expected
        util.check_block_size_params_names_list(block_size_names, tune_params)

        # ensure there is always at least three names
        util.append_default_block_size_names(block_size_names)

        problemsize = self.problemsize_from_gridsizes(kernel_spec["GlobalSize"])
        #lang = None
        lang = "CUPY"

        self.opts = {
            "kernel_name": kernel_spec["KernelName"],
            "kernel_source": self.get_kernel_string(),
            "problem_size": problemsize,
            "arguments": self.manager.arg_handler.populate_args(kernel_spec["Arguments"]),
            "lang": lang,
            "tune_params": tune_params,
            "atol": 1e-6,
            "iterations": iterations,
            "verbose": False,
            "objective":"time",
            "device": 0,
            "platform": 0,
            #"smem_args": None,
            #"cmem_args": None,
            #"texmem_args": None,
            "compiler_options": kernel_spec["CompilerOptions"],
            "block_size_names": block_size_names,
            "quiet": True,
        }

        # create KernelSource
        self.kernelsource = kernel_tuner.core.KernelSource(self.opts["kernel_name"], self.opts["kernel_source"], lang=lang, defines=None)

        # create option bags
        self.kernel_options = Options([(k, self.opts.get(k, None)) for k in _kernel_options.keys()])
        self.tuning_options = Options([(k, self.opts.get(k, None)) for k in _tuning_options.keys()])
        self.device_options = Options([(k, self.opts.get(k, None)) for k in _device_options.keys()])

        self.tuning_options.cachefile = None


    def get_kernel_string(self) -> str:
        """ Reads in the kernel as a string """
        kernel_string = ""
        with open(get_kernel_path(self.spec), 'r') as f:
            kernel_string += f.read()
        return kernel_string

    def problemsize_from_gridsizes(self, gridsizes: dict):
        """ Takes the grid sizes and returns the problem size as a lambda function """
        problemsizes = list()
        dimensions = ['X', 'Y',
                      'Z']  # order in which Kernel Tuner expects the tuple
        for dimension in dimensions:
            if dimension not in gridsizes:
                break
            gridsize = gridsizes[dimension]
            # not an expression, so must be a scalar
            if not isinstance(gridsize, str):
                problemsizes.append(gridsize)
                continue
            # gridsize is expression, so find and wrap the variables
            paramnames = [
                node.id for node in ast.walk(ast.parse(gridsize))
                if isinstance(node, ast.Name)
            ]
            for paramname in paramnames:
                if not f"p['{paramname}']" in gridsize:  # prevent multiple occurances of the same parameter name in the same gridsize from applying this twice
                    gridsize = gridsize.replace(paramname, f"p['{paramname}']")
            problemsizes.append(gridsize)

        # check if the strict use of X, Y, Z causes problems
        if len(problemsizes) != len(gridsizes.keys()):
            raise ValueError(
                f"The problem size dimensions ({dimensions}) do not match the gridsizes specified ({gridsizes})"
            )

        # return the lambda function for future evaluation
        return lambda p: tuple(
            eval(ps, dict(p=p)) if isinstance(ps, str) else ps
            for ps in problemsizes)
        return lam




    def convert_results(self, kt):
        cache = kt["cache"]
        results = []
        time_names = ("verification_time", "compile_time", "times", "time")
        for conf in cache.values():
            new_conf = {}
            new_times = {}
            for (key, value) in conf.items():
                if key in time_names:
                    if key == "times":
                        new_times["runtimes"] = [v/1000 for v in value]
                    elif key == "time":
                        new_times["runtime"] = value / 1000
                    else:
                        new_times[key] = value / 1000
                else:
                    new_conf[key] = value
            results.append(Result(config=new_conf, times=new_times, objective=new_times["runtime"]))
        return results


    def get_results(self, cache_path):
        with open(f"{cache_path}.json", 'r') as f:
            kt = json.loads(f.read())
        return self.convert_results(kt)

    def invalid_result(self, result, msg, error=None):
        result.validity = msg
        result.correctness = 0
        result.runtimes = [0]
        if error:
            result.error = error
        return result

    def generate_compiler_options(self, tuning_config, result):
        benchmark_config = self.spec.get("BenchmarkConfig", {})
        compiler_options = self.kernel_spec.get("CompilerOptions", [])
        for (key, val) in tuning_config.items():
            compiler_options.append(f"-D{key}={val}")
        #for (key, val) in benchmark_config.items():
        #    compiler_options.append(f"-D{key}={val}")
        for (key, val) in result.launch.items():
            compiler_options.append(f"-D{key}={val}")
        return compiler_options

    def get_context(self):
        return self.context

    def get_launch_config(self):
        kernel_spec = self.spec["KernelSpecification"]

        launch_config = {
            "GRID_SIZE_X": eval(str(kernel_spec["GlobalSize"].get("X", 1)), self.get_context()),
            "GRID_SIZE_Y": eval(str(kernel_spec["GlobalSize"].get("Y", 1)), self.get_context()),
            "GRID_SIZE_Z": eval(str(kernel_spec["GlobalSize"].get("Z", 1)), self.get_context()),
            "BLOCK_SIZE_X": eval(str(kernel_spec["LocalSize"].get("X", 1)), self.get_context()),
            "BLOCK_SIZE_Y": eval(str(kernel_spec["LocalSize"].get("Y", 1)), self.get_context()),
            "BLOCK_SIZE_Z": eval(str(kernel_spec["LocalSize"].get("Z", 1)), self.get_context()),
        }

        global_size_type = kernel_spec.get("GlobalSizeType", "CUDA")
        if global_size_type.lower() == "opencl":
            launch_config["GRID_SIZE_X"] = math.ceil(launch_config["GRID_SIZE_X"]/launch_config["BLOCK_SIZE_X"])
            launch_config["GRID_SIZE_Y"] = math.ceil(launch_config["GRID_SIZE_Y"]/launch_config["BLOCK_SIZE_Y"])
            launch_config["GRID_SIZE_Z"] = math.ceil(launch_config["GRID_SIZE_Z"]/launch_config["BLOCK_SIZE_Z"])

        self.add_to_context(launch_config)

        if kernel_spec.get("SharedMemory"):
            launch_config["SHARED_MEMORY_SIZE"] = eval(str(kernel_spec["SharedMemory"]), self.get_context())

        self.grid_dim = (launch_config["GRID_SIZE_X"], launch_config["GRID_SIZE_Y"], launch_config["GRID_SIZE_Z"])
        self.block_dim = (launch_config["BLOCK_SIZE_X"], launch_config["BLOCK_SIZE_Y"], launch_config["BLOCK_SIZE_Z"])
        self.shared_mem_size = launch_config.get("SHARED_MEMORY_SIZE", 0)
        return launch_config

    def reset_context(self):
        self.context = {}
        self.kernel_spec["CompilerOptions"] = copy.deepcopy(self.original_compiler_options)


    def add_to_context(self, d):
        self.context.update(d)


    def run(self, tuning_config, result):

        self.reset_context()
        self.tuning_config = tuning_config
        self.add_to_context(self.spec["BenchmarkConfig"])
        self.add_to_context(self.tuning_config)

        result.launch = self.get_launch_config()
        searchspace = [ tuning_config.values() ]

        #print(f"Config: {tuning_config}")
        try:
            self.kernel_options["compiler_options"] = self.generate_compiler_options(tuning_config, result)

             # create runner
            self.runner = SequentialRunner(self.kernelsource, self.kernel_options, self.device_options, self.opts["iterations"], None)
            self.runner.warmed_up = True # disable warm up for this test
            results, _ = self.runner.run(searchspace, self.kernel_options, self.tuning_options)
            kt_result = results[0]
            result.runtimes = [t/1000 for t in kt_result["times"]]
            result.objective = kt_result["time"]/1000
            result.compile_time = kt_result["compile_time"]/1000
            #result.time = kt_result["verification_time"]
            #result.time = kt_result["benchmark_time"]
            result.algorithm_time = kt_result["strategy_time"]/1000
            result.framework_time = kt_result["framework_time"]/1000
        except Exception as e:
            #print(e)
            return self.invalid_result(result, "Compile exception", e)
        #print(result)
        self.reset_context()
        return result

