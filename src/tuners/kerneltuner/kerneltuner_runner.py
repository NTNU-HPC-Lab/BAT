from __future__ import print_function

from builtins import str

import kernel_tuner
from pathlib import Path
from collections import OrderedDict

from src.reader import reader, T1_specification


class KernelTuner:

    def main(self, args):
        self.prog_args = args
        self.spec = T1_specification.get_spec(args.json)
        strategy_options = dict(popsize=0, max_fevals=10)
        self.tune(args.device_name, strategy_options=strategy_options)

    # def make_gridsize_lambda(x, y, z):

    def tune(self, device_name, strategy="mls", strategy_options=None, verbose=True, quiet=False, simulation_mode=False):
        config_space = self.spec["configurationSpace"]
        kernel_spec = self.spec["kernelSpecification"]
        kernel_name = kernel_spec["kernelName"]

        # read in kernel
        kernel_string = ""
        path = Path(__file__).parent / "../../benchmarks" / kernel_spec["benchmarkName"] / kernel_spec["kernelFile"]
        with open(path, "r") as fp:
            kernel_string += fp.read()

        # get arguments
        args = T1_specification.populate_args(kernel_spec)

        # get problem-, block-, thread-, and grid sizes
        problem_size = tuple(s for s in kernel_spec["gridSize"].values())
        problem_size = lamda p: (73728//p["BLOCK_SIZE"]//p["WORK_PER_THREAD"], 1, 1)
        block_size_names = list(n for n in kernel_spec["blockSize"] if isinstance(n, str))
        grid_div_x = None
        grid_div_y = None
        # mods = kernel_spec["modifiers"]
        # if mods and mods["type"] == "block_size" and mods["action"] == 'divide':
        #     if mods["dimension"] == "X":
        #         grid_div_x = list([mods["parameter"]])
        #     elif mods["dimension"] == "Y":
        #         grid_div_y = list([mods["parameter"]])
        #     else:
        #         print(f"Unkown modifier dimension {mods['dimension']}")

        # get grid divisors
        # can be callable!

        # add tune params
        tune_params = OrderedDict()
        for param in config_space["tuningParameters"]:
            tune_params[param["name"]] = eval(str(param["values"]))

        # add restrictions
        restrict = []
        # for constraint in self.spec["constraints"]:
        #     restrict += constraint

        results, env = kernel_tuner.tune_kernel(kernel_name, kernel_string, problem_size, args, tune_params, lang='cupy', block_size_names=block_size_names,
                                                restrictions=restrict, verbose=verbose, quiet=quiet, grid_div_x=grid_div_x, grid_div_y=grid_div_y, device=0,
                                                platform=0, iterations=32, cache="BAT_" + kernel_name + "_" + device_name, strategy=strategy,
                                                strategy_options=strategy_options, simulation_mode=simulation_mode)
