from builtins import str
import ast
import json
import pandas as pd
from collections import OrderedDict

import kernel_tuner

from src.manager import Manager
from src.result import Result


class KernelTuner:

    def main(self, args):
        self.prog_args = args
        self.manager = Manager(args)
        self.f_evals = self.manager.budget_trials
        strategy_options = dict(max_fevals=self.f_evals)
        return self.tune(args.gpu_name, strategy_options=strategy_options)

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


    def convert_results(self, kt):
        cache = kt["cache"]
        results = []
        time_names = ("verification_time", "compile_time", "times", "time")
        for conf in cache.values():
            new_conf = {}
            new_times = {}
            invalid_conf = False
            for (key, value) in conf.items():
                if key in time_names:
                    if key == "times":
                        new_times["runtimes"] = [v/1000 for v in value]
                    elif key == "time":
                        if isinstance(value, str):
                            error_msg = value
                            new_times["runtime"] = error_msg
                            invalid_conf = True
                        else:
                            new_times["runtime"] = value / 1000
                    else:
                        new_times[key] = value / 1000
                else:
                    new_conf[key] = value
            if invalid_conf:
                invalid_conf = False
                results.append(self.invalid_result(Result(config=new_conf), new_times["runtime"]))
            else:
                results.append(Result(config=new_conf, times=new_times, objective=new_times["runtime"]))
        return results


    def get_results(self, cache_path):
        with open(f"{cache_path}.json", 'r') as f:
            kt = json.loads(f.read())
        return self.convert_results(kt)


    def run_tune(self, gpu_name, strategy, strategy_options, verbose, quiet, simulation_mode):
        kernel_spec = self.manager.spec["KernelSpecification"]
        kernel_name = kernel_spec["KernelName"]
        language = kernel_spec["Language"]
        if language != "CUDA":
            raise NotImplementedError(
                "Currently only CUDA kernels have been implemented")

        # read in kernel
        kernel_string = self.manager.runner.get_kernel_string()

        # get arguments
        args, cmem_args = self.manager.arg_handler.populate_args(kernel_spec["Arguments"])
        iterations = eval(
            str(self.manager.spec["BenchmarkConfig"]["iterations"])
        )  # number of times each kernel configuration is ran
        compiler_options = kernel_spec["CompilerOptions"]
        # precision = self.spec["benchmarkConfig"]["PRECISION"]    # whether to use single or double precision (encoded as 32 or 64)

        # get problem-, block-, thread-, and grid sizes
        if kernel_spec.get("ProblemSize"):
            ps = kernel_spec["ProblemSize"]
            problem_size = ps if isinstance(ps, int) else tuple(ps)

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
        results, env = kernel_tuner.tune_kernel(
            kernel_name,
            kernel_string,
            problem_size,
            args,
            tune_params,
            cmem_args=cmem_args,
            lang='cupy',
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
            simulation_mode=simulation_mode)



    def tune(self,
             gpu_name,
             strategy="random_sample",
             strategy_options=None,
             verbose=True,
             quiet=False,
             simulation_mode=False):
        if self.prog_args.cache:
            self.cache_path = self.prog_args.cache
        else:
            self.cache_path = f"BAT_{self.manager.dataset.hash}"
            self.run_tune(gpu_name, strategy, strategy_options, verbose, quiet, simulation_mode)

        results = self.get_results(self.cache_path)
        self.manager.dataset.write_interval = len(results)
        for res in results:
            self.manager.dataset.add_result(res)

        #self.manager.write()
        #self.manager.dataset.copy_and_delete_file(filepath=f"{cache_path}.json", filename="KT_cache.json")
        self.manager.dataset.final_write_data()


        best = self.manager.dataset.get_best()
        self.manager.finished()
        return best
