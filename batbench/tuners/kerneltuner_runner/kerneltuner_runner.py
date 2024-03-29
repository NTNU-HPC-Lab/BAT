from collections import OrderedDict
import ast
import json

import kernel_tuner
from kernel_tuner.util import (ErrorConfig)

from batbench.manager.manager import Manager
from batbench.result.result import Result


class KernelTuner:

    def main(self, args):
        self.prog_args = args
        self.manager = Manager(args)
        self.f_evals = self.manager.budget_trials
        DEFAULT_OBJECTIVE = "time"
        self.objective = self.manager.problem.spec['General'].get('Objective', DEFAULT_OBJECTIVE)
        self.minimize = self.manager.problem.spec['General'].get('Minimize', True)
        strategy_options = {"max_fevals": self.f_evals}
        return self.tune(args.gpu_name, strategy_options=strategy_options)

    def problemsize_from_gridsizes(self, gridsizes: dict):
        """ Takes the grid sizes and returns the problem size as a lambda function """
        problemsizes = []
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
                # prevent multiple occurances of the same parameter name in the same gridsize
                # from applying this twice
                if not f"p['{paramname}']" in gridsize:
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


    def convert_results(self, cache):
        results = []
        time_names = ("verification_time", "compile_time", "benchmark_time",
                      "strategy_time", "framework_time", "times", "time")
        for conf in cache:
            new_conf = {}
            kt_result = conf

            for (key, value) in conf.items():
                if key not in time_names:
                    new_conf[key] = value

            result = Result(config=new_conf)

            if isinstance(kt_result[self.objective], ErrorConfig):
                results.append(self.invalid_result(result, "Compile exception"))
                continue

            unit = 1000 # convert from ms to seconds

            result.runtimes = [t/unit for t in kt_result["times"]]
            result.runtime = sum(result.runtimes)
            result.objective = kt_result[self.objective]
            if self.objective == "time":
                result.objective /= 1000
            result.compile_time = kt_result["compile_time"]/unit
            result.framework_time = kt_result["framework_time"]/unit
            result.algorithm_time = kt_result["strategy_time"]/unit
            result.total_time = sum([result.compile_time, result.framework_time,
                                    result.algorithm_time, result.runtime])
            results.append(result)

        return results


    def get_results(self, cache_path):
        with open(f"{cache_path}.json", 'r', encoding='utf-8') as file:
            kt_cache = json.loads(file.read())
        return self.convert_results(kt_cache["cache"])


    def run_tune(self, gpu_name, strategy, strategy_options, verbose, quiet, simulation_mode):
        kernel_spec = self.manager.problem.spec["KernelSpecification"]
        kernel_name = kernel_spec["KernelName"]
        language = kernel_spec["Language"]
        if language != "CUDA":
            raise NotImplementedError(
                "Currently only CUDA kernels have been implemented")

        # read in kernel
        kernel_string = self.manager.problem.runner.get_kernel_string()

        # get arguments
        args, cmem_args = self.manager.problem.get_args().get_function_args()
        cmem_args = cmem_args if cmem_args else None
        iterations = eval(
            str(self.manager.problem.spec["BenchmarkConfig"]["iterations"])
        )  # number of times each kernel configuration is ran
        compiler_options = kernel_spec["CompilerOptions"]

        # whether to use single or double precision (encoded as 32 or 64)
        # precision = self.spec["benchmarkConfig"]["PRECISION"]

        # get problem-, block-, thread-, and grid sizes
        if kernel_spec.get("ProblemSize"):
            problem_size_spec = kernel_spec["ProblemSize"]
            if isinstance(problem_size_spec, int):
                problem_size = problem_size_spec
            else:
                problem_size = tuple(problem_size_spec)

            grid_div_x = kernel_spec["GridDivX"]
            grid_div_y = kernel_spec["GridDivY"]
        else:
            problem_size = self.problemsize_from_gridsizes(kernel_spec["GlobalSize"])
            grid_div_x = []
            grid_div_y = []

        block_size_names = list(n for n in kernel_spec["LocalSize"].values()
                                if not n.isdigit())

        # add tune params
        tune_params = OrderedDict(self.manager.config_space.get_parameters())

        # add restrictions
        constraints = self.manager.config_space.get_constraints()
        restrict = [c["Expression"] for c in constraints]

        self.results, env = kernel_tuner.tune_kernel(
            kernel_name,
            kernel_string,
            problem_size,
            args,
            tune_params,
            cmem_args=cmem_args,
            lang=self.manager.problem.cuda_backend if self.manager.problem.cuda_backend else 'cupy',
            block_size_names=block_size_names,
            restrictions=restrict,
            verbose=verbose,
            quiet=quiet,
            grid_div_x=grid_div_x,
            grid_div_y=grid_div_y,
            device=0,
            platform=0,
            iterations=iterations,
            cache=self.cache_path,
            compiler_options=compiler_options,
            strategy=strategy,
            strategy_options=strategy_options,
            simulation_mode=simulation_mode,
            metrics=self.manager.problem.metrics)


    def tune(self,
             gpu_name,
             strategy="random_sample",
             strategy_options=None,
             verbose=False,
             quiet=True,
             simulation_mode=False):
        if self.prog_args.cache:
            self.cache_path = self.prog_args.cache
        else:
            #self.cache_path = None
            self.cache_path = f"{self.manager.dataset.dataset_folder}/BAT_{self.manager.dataset.hash}"
            self.run_tune(gpu_name, strategy, strategy_options, verbose, quiet, simulation_mode)


        results = self.convert_results(self.results)

        self.manager.dataset.write_interval = len(results)
        for res in results:
            self.manager.dataset.add_result(res)

        self.manager.dataset.final_write_data()

        best = self.manager.dataset.get_best()
        self.manager.finished()
        return best
