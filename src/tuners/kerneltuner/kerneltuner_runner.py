from __future__ import print_function

from builtins import str
import ast

import kernel_tuner
from pathlib import Path
from collections import OrderedDict

from src.readers.python.cuda.cupy_reader import CupyReader


class KernelTuner:

    def main(self, args):
        self.prog_args = args
        self.reader = CupyReader(args.json)
        self.spec = self.reader.spec
        strategy_options = dict(popsize=0, max_fevals=10)
        self.tune(args.gpu_name, strategy_options=strategy_options)

    def problemsize_from_gridsizes(self, gridsizes: dict):
        """ Takes the grid sizes and returns the problem size as a lambda function """
        problemsizes = list()
        dimensions = ['X', 'Y', 'Z']    # order in which Kernel Tuner expects the tuple
        for dimension in dimensions:
            if dimension not in gridsizes:
                break
            gridsize = gridsizes[dimension]
            # not an expression, so must be a scalar
            if not isinstance(gridsize, str):
                problemsizes.append(gridsize)
                continue
            # gridsize is expression, so find and wrap the variables
            paramnames = [node.id for node in ast.walk(ast.parse(gridsize)) if isinstance(node, ast.Name)]
            for paramname in paramnames:
                gridsize = gridsize.replace(paramname, f"p['{paramname}']")
            problemsizes.append(gridsize)

        # check if the strict use of X, Y, Z causes problems
        if len(problemsizes) != len(gridsizes.keys()):
            raise ValueError(f"The problem size dimensions ({dimensions}) do not match the gridsizes specified ({gridsizes})")

        # return the lambda function for future evaluation
        return lambda p: tuple(eval(ps, dict(p=p)) if isinstance(ps, str) else ps for ps in problemsizes)

    def tune(self, gpu_name, strategy="mls", strategy_options=None, verbose=True, quiet=False, simulation_mode=False):
        config_space = self.spec["configurationSpace"]
        kernel_spec = self.reader.kernel_spec
        kernel_name = kernel_spec["kernelName"]

        # read in kernel
        kernel_string = ""
        path = Path(__file__).parent / "../../benchmarks" / kernel_spec["benchmarkName"] / kernel_spec["kernelFile"]
        with open(path, "r") as fp:
            kernel_string += fp.read()

        # get arguments
        args = self.reader.populate_args()

        # get problem-, block-, thread-, and grid sizes
        problem_size = self.problemsize_from_gridsizes(kernel_spec["gridSize"])
        block_size_names = list(n for n in kernel_spec["blockSize"].values() if isinstance(n, str))
        grid_div_x = []
        grid_div_y = []
        # mods = kernel_spec["modifiers"]
        # if mods and mods["type"] == "block_size" and mods["action"] == 'divide':
        #     if mods["dimension"] == "X":
        #         grid_div_x = list([mods["parameter"]])
        #     elif mods["dimension"] == "Y":
        #         grid_div_y = list([mods["parameter"]])
        #     else:
        #         print(f"Unkown modifier dimension {mods['dimension']}")

        # add tune params
        tune_params = OrderedDict()
        for param in config_space["tuningParameters"]:
            tune_params[param["name"]] = eval(str(param["values"]))

        # add restrictions
        if "constraints" in self.spec:
            for constraint in self.spec["constraints"]:
                restrict += constraint

        results, env = kernel_tuner.tune_kernel(kernel_name, kernel_string, problem_size, args, tune_params, lang='cupy', block_size_names=block_size_names,
                                                restrictions=restrict, verbose=verbose, quiet=quiet, grid_div_x=grid_div_x, grid_div_y=grid_div_y, device=0,
                                                platform=0, iterations=32, cache="BAT_" + kernel_name + "_" + gpu_name, strategy=strategy,
                                                strategy_options=strategy_options, simulation_mode=simulation_mode)
