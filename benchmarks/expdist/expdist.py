from src.config_space.cuda_problem import CUDAProblem
from src.kernelbackend.kerneltuner_runner import KernelBackend
from src.config_space import ConfigSpace
import numpy as np
import os

class Expdist(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("expdist", self.setup_spec())
        self.setup()
        self.run_settings = run_settings
        self.runner = KernelBackend(self.spec, self.config_space, [self.function_args, []], "CUDA")

    def run(self, tuning_config, result):
        return self.runner.run(tuning_config, result)

    def setup(self):
        self.setup_config_space()
        self.setup_args()

    def setup_spec(self):
        spec = {}
        kernel_spec = {}
        kernel_spec["LocalSize"] = {
            "X": "block_size_x",
            "Y": "block_size_y",
#            "Z": "block_size_z"
        }
        size = 32 * 1024
        kernel_spec["GridDivX"] = ["block_size_x", "tile_size_x"]
        kernel_spec["GridDivY"] = ["block_size_y", "tile_size_y"]
        kernel_spec["ProblemSize"] = (size, size, 1)
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
            .sort_parameters()
        )

    def setup_args(self):
        alloc_size = 32*1024
        size = np.int32(32*1024)
        max_blocks = np.int32( np.ceil(size / float(np.amin(self.config_space.get_parameters()["block_size_x"]))) *
                              np.ceil(size / float(np.amin(self.config_space.get_parameters()["block_size_y"]))) )
        ndim = np.int32(2)
        A = np.random.randn(alloc_size*ndim).astype(np.float32)
        B = A+0.00001*np.random.randn(alloc_size*ndim).astype(np.float32)
        scale_A = np.absolute(0.01*np.random.randn(alloc_size).astype(np.float32))
        scale_B = np.absolute(0.01*np.random.randn(alloc_size).astype(np.float32))
        cost = np.zeros((max_blocks)).astype(np.float32)

        self.function_args = [A, B, size, size, scale_A, scale_B, cost]
        return self.function_args

