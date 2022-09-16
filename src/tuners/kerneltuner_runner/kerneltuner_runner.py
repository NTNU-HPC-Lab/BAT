from builtins import str
import ast
from collections import OrderedDict

import kernel_tuner

from src.manager import Manager


class KernelTuner:

    def main(self, args):
        self.prog_args = args
        self.manager = Manager(args)
        strategy_options = dict(popsize=0, max_fevals=10)
        self.tune(args.gpu_name, strategy_options=strategy_options)

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

    def tune(self,
             gpu_name,
             strategy="mls",
             strategy_options=None,
             verbose=True,
             quiet=False,
             simulation_mode=False):
        kernel_spec = self.manager.spec["kernelSpecification"]
        kernel_name = kernel_spec["kernelName"]
        language = kernel_spec["language"]
        if language != "CUDA":
            raise NotImplementedError(
                "Currently only CUDA kernels have been implemented")

        # read in kernel
        kernel_string = self.manager.runner.get_kernel_string()

        # get arguments
        args = self.manager.arg_handler.populate_args(kernel_spec["arguments"])
        iterations = eval(
            str(self.manager.spec["benchmarkConfig"]["iterations"])
        )  # number of times each kernel configuration is ran
        compiler_options = kernel_spec["compilerOptions"]
        # precision = self.spec["benchmarkConfig"]["PRECISION"]    # whether to use single or double precision (encoded as 32 or 64)

        # get problem-, block-, thread-, and grid sizes
        problem_size = self.problemsize_from_gridsizes(kernel_spec["gridSize"])
        block_size_names = list(n for n in kernel_spec["blockSize"].values()
                                if isinstance(n, str))
        grid_div_x = []
        grid_div_y = []

        # add tune params
        tune_params = OrderedDict(self.manager.config_space.get_parameters())

        # add restrictions
        restrict = self.manager.config_space.get_constraints()

        results, env = kernel_tuner.tune_kernel(
            kernel_name,
            kernel_string,
            problem_size,
            args,
            tune_params,
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
            cache="BAT_" + kernel_name + "_" + gpu_name,
            compiler_options=compiler_options,
            strategy=strategy,
            strategy_options=strategy_options,
            simulation_mode=simulation_mode)
