from collections import OrderedDict
from src.config_space.cuda_problem import CUDAProblem
from src.kernelbackend.kerneltuner_runner import KernelBackend
from src.config_space import ConfigSpace
import numpy as np
import os

class Expdist(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("expdist", self.setup_spec())
        self.run_settings = run_settings
        self.setup()
        self.runner = KernelBackend(
            self.spec, self.config_space, 
            self.function_args, cuda_backend="CUDA", 
            metrics=self.metrics,
            objective=self.objective
        )

    def run(self, tuning_config, result):
        return self.runner.run(tuning_config, result)

    def setup(self):
        self.setup_config_space()
        self.setup_args()
        self.setup_objective_and_metrics()

    def setup_objective_and_metrics(self):
        self.objective = "GFLOPs"
        metrics = OrderedDict()

        def FLOPs_in_partial_reduction(p):
            num_thread_blocks = np.ceil(self.size/(p["block_size_x"]*p["tile_size_x"])) * np.ceil(self.size/(p["block_size_y"]*p["tile_size_y"]))
            ops_per_thread_block = p["block_size_x"]*p["block_size_y"]/32*31+31 #minimal number of ops per warp times number of warps + #ops for 1 final warp
            return num_thread_blocks*ops_per_thread_block

        ops_per_iteration = 35 #from Nsight profiler
        metrics[self.objective] = lambda p: ((FLOPs_in_partial_reduction(p)+ops_per_iteration*self.size*self.size) /1e9) / (p["time"] / 1e3)
        self.metrics = metrics

    def setup_spec(self):
        spec = {}
        kernel_spec = {}
        kernel_spec["LocalSize"] = {
            "X": "block_size_x",
            "Y": "block_size_y",
#            "Z": "block_size_z"
        }
        self.size = 32 * 1024
        kernel_spec["GridDivX"] = ["block_size_x", "tile_size_x"]
        kernel_spec["GridDivY"] = ["block_size_y", "tile_size_y"]
        kernel_spec["ProblemSize"] = (self.size, self.size, 1)
        kernel_spec["Language"] = "CUDA"
        kernel_spec["KernelName"] = "ExpDist"
        kernel_spec["CompilerOptions"] = []
        kernel_spec["KernelFile"] = "expdist.cu"

        spec["KernelSpecification"] = kernel_spec
        spec["General"] = {}
        spec["General"]["BenchmarkName"] = "expdist"
        spec["BenchmarkConfig"] = {}
        spec["BenchmarkConfig"]["iterations"] = 10 # TODO: Fix

        self.spec = spec
        return self.spec


    def setup_config_space(self):
        self.config_space = (ConfigSpace()
            .add_enum("block_size_x", [2**i for i in range(5,11)][::-1])
            .add_enum("block_size_y", [1])
            .add_enum("tile_size_x", [i for i in range(1, 9)])
            .add_enum("tile_size_y", [1])
            .add_enum("use_shared_mem", [0, 1, 2])
            .add_enum("loop_unroll_factor_x", [1])
            .add_enum("loop_unroll_factor_y", [1])
            .add_enum("use_column", [0])
            .add_enum("n_y_blocks", [1])
            .add_constraint("not(use_column == 0 and n_y_blocks > 1)", ["use_column", "n_y_blocks"])
            .add_constraint("not(use_column == 0 and use_shared_mem == 2)", ["use_column", "use_shared_mem"])
            .add_constraint("not(loop_unroll_factor_x > tile_size_x or (loop_unroll_factor_x > 0 and tile_size_x % loop_unroll_factor_x != 0))", ["loop_unroll_factor_x", "tile_size_x"])
        )

    def setup_args(self):
        alloc_size = self.size
        size = np.int32(self.size)
        max_blocks = np.int32( np.ceil(size / float(np.amin(self.config_space.get_parameters()["block_size_x"]))) *
                              np.ceil(size / float(np.amin(self.config_space.get_parameters()["block_size_y"]))) )
        ndim = np.int32(2)
        A = np.random.randn(alloc_size*ndim).astype(np.float32)
        B = A + 0.00001*np.random.randn(alloc_size*ndim).astype(np.float32)
        scale_A = np.absolute(0.01*np.random.randn(alloc_size).astype(np.float32))
        scale_B = np.absolute(0.01*np.random.randn(alloc_size).astype(np.float32))
        cost = np.zeros((max_blocks)).astype(np.float32)

        self.function_args = [[A, B, size, size, scale_A, scale_B, cost], []]
        return self.function_args

