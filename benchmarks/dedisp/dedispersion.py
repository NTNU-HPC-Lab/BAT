from src.config_space.cuda_problem import CUDAProblem
from src.kernelbackend.kerneltuner_runner import KernelBackend
from src.config_space import ConfigSpace
import numpy as np
import os

nr_dms = 2048
nr_samples = 25000
nr_channels = 1536
max_shift = 650
nr_samples_per_channel = (nr_samples+max_shift)
down_sampling = 1
dm_first = 0.0
dm_step = 0.02

channel_bandwidth = 0.1953125
sampling_time = 0.00004096
min_freq = 1425.0
max_freq = min_freq + (nr_channels-1) * channel_bandwidth


class Dedispersion(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("dedispersion", self.setup_spec())
        self.setup()
        self.run_settings = run_settings
        self.runner = KernelBackend(self.spec, self.config_space, [self.function_args, []])

    def run(self, tuning_config, result):
        return self.runner.run(tuning_config, result)

    def setup(self):
        self._create_reference()
        self.setup_config_space()
        self.setup_args()

    def setup_spec(self):
        spec = {}
        kernel_spec = {}
        kernel_spec["LocalSize"] = {
            "X": "block_size_x",
            "Y": "block_size_y",
            "Z": "block_size_z"
        }
        kernel_spec["GridDivX"] = None
        kernel_spec["GridDivY"] = None
        kernel_spec["ProblemSize"] = (nr_samples, nr_dms, 1)
        kernel_spec["Language"] = "CUDA"
        kernel_spec["KernelName"] = "dedispersion_kernel"
        kernel_spec["CompilerOptions"] = [f"-I{os.path.dirname(os.path.realpath(__file__))}"]
        kernel_spec["KernelFile"] = "dedispersion.cu"

        spec["KernelSpecification"] = kernel_spec
        spec["General"] = {}
        spec["General"]["BenchmarkName"] = "dedisp"
        spec["BenchmarkConfig"] = {}
        spec["BenchmarkConfig"]["iterations"] = 10 # TODO: Fix

        self.spec = spec
        return self.spec


    def setup_config_space(self):
        self.config_space = (ConfigSpace()
            .add_enum("block_size_x", [1, 2, 4, 8] + [16*i for i in range(1,33)])
            .add_enum("block_size_y", [4*i for i in range(1,33)])
            .add_enum("block_size_z", [1])
            .add_enum("tile_size_x", [i for i in range(1,17)])
            .add_enum("tile_size_y", [i for i in range(1,17)])
            .add_enum("tile_stride_x", [0, 1])
            .add_enum("tile_stride_y", [0, 1])
            .add_enum("loop_unroll_factor_channel", [0] + [i for i in range(1,nr_channels+1) if nr_channels % i == 0])
            .add_enum("blocks_per_sm", [i for i in range(5)])

            .add_constraint("32 <= block_size_x * block_size_y <= 1024", ["block_size_x", "block_size_y"])
            .add_constraint("tile_size_x > 1 or tile_stride_x == 0", ["tile_size_x", "tile_stride_x"])
            .add_constraint("tile_size_y > 1 or tile_stride_y == 0", ["tile_size_y", "tile_stride_y"])

            #.add_constraint("loop_unroll_factor_x <= tile_size_x and tile_size_x % loop_unroll_factor_x == 0")
            #.add_constraint("loop_unroll_factor_y <= tile_size_y and tile_size_y % loop_unroll_factor_y == 0")
            #.add_constraint(f"loop_unroll_factor_channel <= {nr_channels} and loop_unroll_factor_channel and {nr_channels} % loop_unroll_factor_channel == 0")
            .sort_parameters()
        )

    def setup_args(self):
        input_samples = np.random.randn(nr_samples_per_channel*nr_channels).astype(np.uint8)

        output_arr = np.zeros(nr_dms*nr_samples, dtype=np.float32)
        shifts = self._get_shifts()
        self.function_args = [input_samples, output_arr, shifts]
        return self.function_args

    def _get_shifts(self):
        max_freq = min_freq + ((nr_channels - 1) * channel_bandwidth)
        inverse_high_freq = 1/max_freq**2
        time_unit = nr_samples * sampling_time

        channels = np.arange(nr_channels, dtype=np.float32)
        inverse_freq = 1.0 / (min_freq + (channels * channel_bandwidth))**2

        # 4148.808 is the time delay per dispersion measure, a constant in the dispersion equation
        shifts_float = (4148.808 * (inverse_freq - inverse_high_freq) * (nr_samples / down_sampling)) / time_unit
        shifts_float[-1] = 0
        return shifts_float

    def _create_reference(self):

        input_samples = np.random.randn(nr_samples_per_channel*nr_channels).astype(np.uint8)
        shifts = self._get_shifts()

        np.save("input_ref", input_samples, allow_pickle=False)
        np.save("shifts_ref", shifts, allow_pickle=False)
