from builtins import str
import ast
import json
import pandas as pd
import numpy
import copy
from collections import OrderedDict
import traceback

import kernel_tuner
from kernel_tuner.interface import Options, _kernel_options, _device_options, _tuning_options
from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner import util
from kernel_tuner.util import (ErrorConfig)

from src.manager import get_kernel_path
from src.manager import Manager
from src.result import Result


class KernelBackend:
    def __init__(self, args, manager):
        self.prog_args = args
        self.manager = manager
        self.spec = manager.spec
        self.kernel_spec = self.spec["KernelSpecification"]
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

        # check for forbidden names in tune parameters
        util.check_tune_params_list(tune_params, observers)

        # check whether block_size_names are used as expected
        util.check_block_size_params_names_list(block_size_names, tune_params)

        # ensure there is always at least three names
        util.append_default_block_size_names(block_size_names)

        if kernel_spec.get("ProblemSize"):
            ps = kernel_spec["ProblemSize"]
            problem_size = ps if isinstance(ps, int) else tuple(ps)

            grid_div_x = kernel_spec["GridDivX"]
            grid_div_y = kernel_spec["GridDivY"]
        else:
            problem_size = self.problemsize_from_gridsizes(kernel_spec["GlobalSize"])
            grid_div_x = []
            grid_div_y = []

        lang = "CUPY"
        args, cmem_args = self.manager.arg_handler.populate_args(kernel_spec["Arguments"])

        debug = False
        verbose = debug
        quiet = not debug

        self.opts = {
            "kernel_name": kernel_spec["KernelName"],
            "kernel_source": self.get_kernel_string(),
            "problem_size": problem_size,
            "arguments": args,
            "lang": lang,
            "tune_params": tune_params,
            "atol": 1e-6,
            "iterations": iterations,
            "verbose": verbose,
            "objective":"time",
            "device": 0,
            "platform": 0,
            "grid_div_x": grid_div_x ,
            "grid_div_y": grid_div_y,
            #"smem_args": None,
            "cmem_args": cmem_args if cmem_args else None,
            #"texmem_args": None,
            "compiler_options": kernel_spec["CompilerOptions"],
            "block_size_names": block_size_names,
            "quiet": quiet,
        }

        # create KernelSource
        self.kernelsource = kernel_tuner.core.KernelSource(self.opts["kernel_name"], self.opts["kernel_source"], lang=lang, defines=None)

        # create option bags
        self.kernel_options = Options([(k, self.opts.get(k, None)) for k in _kernel_options.keys()])
        self.tuning_options = Options([(k, self.opts.get(k, None)) for k in _tuning_options.keys()])
        self.device_options = Options([(k, self.opts.get(k, None)) for k in _device_options.keys()])

        self.tuning_options.cachefile = None

        self.runner = SequentialRunner(self.kernelsource, self.kernel_options, self.device_options, self.opts["iterations"], None)
        self.runner.warmed_up = True # disable warm up for this test


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


    def invalid_result(self, result, msg, error=None):
        result.validity = msg
        result.correctness = 0
        result.runtimes = [0]
        if error:
            result.error = error
        return result


    def run(self, tuning_config, result):

        self.tuning_config = tuning_config
        result.config = tuning_config
        searchspace = [ tuning_config.values() ]

        results, _ = self.runner.run(searchspace, self.kernel_options, self.tuning_options)
        kt_result = results[0]
        if isinstance(kt_result["time"], ErrorConfig):
            return self.invalid_result(result, "Compile exception")
        result.runtimes = [t/1000 for t in kt_result["times"]]
        result.objective = kt_result["time"]/1000
        result.compile_time = kt_result["compile_time"]/1000
        #result.time = kt_result["verification_time"]
        #result.time = kt_result["benchmark_time"]
        #result.algorithm_time = kt_result["strategy_time"]/1000
        result.framework_time = kt_result["framework_time"]/1000
        return result

