import ast
import logging
from collections import OrderedDict
import re

import kernel_tuner
from kernel_tuner.interface import run_kernel, Options
from kernel_tuner.interface import _kernel_options, _device_options, _tuning_options
from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner import util
from kernel_tuner.util import (ErrorConfig)
from batbench.config_space.arguments import Arguments
from batbench.result.result import Result

from batbench.util import get_kernel_path

class KernelBackend:
    CUDA = "CUDA"
    CUPY = "Cupy"
    TIME = "time"
    DEFAULT_ATOL = 1e-6
    DEFAULT_DEVICE = 0
    DEFAULT_PLATFORM = 0
    DEFAULT_OBJECTIVE = TIME

    def __init__(self, spec, config_space, args: Arguments,
                 cuda_backend="Cupy", metrics=None):
        self.spec = spec
        self.config_space = config_space
        self.kernel_spec = self.spec["KernelSpecification"]
        self.objective = self.spec['General'].get('Objective', self.DEFAULT_OBJECTIVE)
        self.minimize = self.spec['General'].get('Minimize', True)
        self.metrics = metrics
        self.args = args
        self.function_args = self.args.get_function_args()

        self.validate_kernel_spec()
        tune_params = OrderedDict(self.config_space.get_parameters())
        block_size_names = self.get_block_size_names()
        print(f"Block_size: {block_size_names}")
        self.validate_params(tune_params, block_size_names)
        self.extend_block_size_names(block_size_names)

        problem_size, grid_div_x, grid_div_y = self.get_problem_size_and_grid_div()
        data_args, cmem_args = self.function_args

        print(f"Block_size: {block_size_names}")
        self.opts = self.create_opts(data_args, tune_params, block_size_names, cmem_args, problem_size, grid_div_x, grid_div_y, cuda_backend)
        self.create_kernelsource(cuda_backend)
        self.create_option_bags()
        self.setup_sequential_runner()

    def validate_kernel_spec(self) -> None:
        """Validate the kernel specification language."""
        if self.kernel_spec["Language"] not in (self.CUDA, "HIP"):
            raise NotImplementedError(
                "Currently only CUDA kernels have been implemented")

    def get_block_size_names(self):
        return [n for n in self.kernel_spec["LocalSize"].values() if not n.isdigit()]

    def validate_params(self, tune_params, block_size_names):
        observers = None
        util.check_tune_params_list(tune_params, observers)
        util.check_block_size_params_names_list(block_size_names, tune_params)

    def extend_block_size_names(self, block_size_names):
        if len(block_size_names) < 3:
            block_size_names.extend(["block_size_z"])
        #util.append_default_block_size_names(block_size_names)

    def get_problem_size_and_grid_div(self):
        if self.kernel_spec.get("ProblemSize"):
            problem_size = self.kernel_spec["ProblemSize"]
            problem_size = problem_size if isinstance(problem_size, int) else tuple(problem_size)
            grid_div_x = self.kernel_spec["GridDivX"]
            grid_div_y = self.kernel_spec["GridDivY"]
        else:
            problem_size = self.problemsize_from_gridsizes(self.kernel_spec["GlobalSize"])
            grid_div_x = []
            grid_div_y = []
        return problem_size, grid_div_x, grid_div_y

    def create_opts(self, args, tune_params, block_size_names, cmem_args,
                    problem_size, grid_div_x, grid_div_y, cuda_backend):
        return {
            "kernel_name": self.kernel_spec["KernelName"],
            "kernel_source": self.get_kernel_string(),
            "problem_size": problem_size,
            "arguments": args,
            "lang": cuda_backend,
            "atol": self.DEFAULT_ATOL,
            "tune_params": tune_params,
            "iterations": eval(str(self.spec["BenchmarkConfig"]["iterations"])),
            "verbose": False,
            "objective": self.objective,
            "device": self.DEFAULT_DEVICE,
            "platform": self.DEFAULT_PLATFORM,
            "grid_div_x": grid_div_x,
            "grid_div_y": grid_div_y,
            "cmem_args": cmem_args if cmem_args else None,
            "compiler_options": self.kernel_spec["CompilerOptions"],
            "block_size_names": block_size_names,
            "quiet": True,
            "metrics": self.metrics,
        }

    def create_kernelsource(self, cuda_backend):
        self.kernelsource = kernel_tuner.core.KernelSource(
            self.opts["kernel_name"], self.opts["kernel_source"], lang=cuda_backend, defines=None)

    def create_option_bags(self):
        """Create options for kernel, tuning and device."""
        self.kernel_options = self.create_options(self.opts, _kernel_options)
        self.tuning_options = self.create_options(self.opts, _tuning_options)
        self.device_options = self.create_options(self.opts, _device_options)
        self.tuning_options.cachefile = None

    def create_options(self, opts, opt_keys):
        """Create an Options object from given opts and opt_keys."""
        return Options([(k, opts.get(k, None)) for k in opt_keys.keys()])

    def setup_sequential_runner(self):
        # Prints the types of arguments
        t = [type(self.kernel_options["arguments"][i]) for i in range(len(self.kernel_options["arguments"]))]
        print(t)
        self.runner = SequentialRunner(self.kernelsource, self.kernel_options,
                                       self.device_options, self.opts["iterations"], None)
        self.runner.warmed_up = True  # disable warm up for this test

    def get_kernel_string(self) -> str:
        """ Reads in the kernel as a string """
        with open(get_kernel_path(self.spec), 'r', encoding='utf-8') as file:
            kernel_string = file.read()
        return kernel_string

    def problemsize_from_gridsizes(self, gridsizes: dict):
        problemsizes = []
        dimensions = ['X', 'Y', 'Z']
        for dimension in dimensions:
            if dimension not in gridsizes:
                break
            problemsizes.append(self.evaluate_gridsize(gridsizes, dimension))
        self.validate_problemsize_length(problemsizes, gridsizes)
        return lambda p: tuple(
            eval(ps, dict(p=p)) if isinstance(ps, str) else ps
            for ps in problemsizes)

    def evaluate_gridsize(self, gridsizes, dimension):
        gridsize = gridsizes[dimension]
        if not isinstance(gridsize, str):
            return gridsize
        paramnames = self.extract_param_names(gridsize)
        return self.wrap_variables_in_gridsize(gridsize, paramnames)

    def extract_param_names(self, gridsize):
        return [node.id for node in ast.walk(ast.parse(gridsize)) if isinstance(node, ast.Name)]


    def wrap_variables_in_gridsize(self, gridsize, paramnames):
        for paramname in paramnames:
            # Using a regular expression to ensure that whole words are matched
            pattern = r'\b' + re.escape(paramname) + r'\b'
            replacement = f"p['{paramname}']"

            # Check if the parameter name is already wrapped
            wrapped_pattern = f"p\\['{paramname}'\\]"
            if not re.search(wrapped_pattern, gridsize):
                gridsize = re.sub(pattern, replacement, gridsize)
        return gridsize

    def validate_problemsize_length(self, problemsizes, gridsizes):
        if len(problemsizes) != len(gridsizes.keys()):
            raise ValueError(
                f"The problem size dimensions (X, Y, Z) do not match the gridsizes specified ({gridsizes})")

    def update_invalid_result(self, result, msg, error=None):
        result.validity = msg
        result.correctness = 0
        result.runtimes = [0]
        result.objective = 10000 if self.minimize else 0
        if error:
            result.error = error
        return result


    def update_result(self, result, kt_result):
        result.runtimes = [t/1000 for t in kt_result["times"]]
        result.runtime = sum(result.runtimes)
        result.objective = kt_result[self.objective]
        if self.objective == self.TIME:
            result.objective /= 1000
        result.compile_time = kt_result["compile_time"]/1000
        #result.time = kt_result["verification_time"]
        #result.time = kt_result["benchmark_time"]
        #result.algorithm_time = kt_result["strategy_time"]/1000
        #result.framework_time = kt_result["framework_time"]/1000

    def run_reference(self, tuning_config):
        res = run_kernel(self.opts["kernel_name"],
                   self.opts["kernel_source"], self.opts["problem_size"],
                   self.opts["arguments"], tuning_config,
                   self.opts["grid_div_x"], self.opts["grid_div_y"], None,
                   self.opts["lang"], self.opts["device"], self.opts["platform"],
                   None, self.opts["cmem_args"], None, None,
                   self.opts["compiler_options"], None, self.opts["block_size_names"],
                   self.opts["quiet"], None)
        answer_list = [None] * len(res)

        for key in self.args.output_args:
            idx = self.args.args[key]["index"]
            self.args.add_reference_value(key, res[idx])
            answer_list[idx] = res[idx]
        self.opts["answer"] = answer_list
        self.create_option_bags()
        print("Finished reference run", answer_list)
        return res

    def run(self, tuning_config, result: Result) -> Result:
        result.config = tuning_config
        searchspace = [ tuning_config.values() ]
        try:
            #results, _ = self.runner.run(searchspace, 
            results = self.runner.run(searchspace, 
                    #self.kernel_options, 
                    self.tuning_options)
        except RuntimeError:
            return self.update_invalid_result(result, "RuntimeError")
        kt_result = results[0]
        if self.objective in kt_result and isinstance(kt_result[self.objective], ErrorConfig):
            logging.error("Failed to run with tuning config: %s. Error: %s",
                          tuning_config, kt_result[self.objective])
            return self.update_invalid_result(result, "Compile exception")
        self.update_result(result, kt_result)
        return result
