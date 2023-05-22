import time
import math
import cupy as cp
import copy

from src.config_space import ConfigSpace

from .arg_handler import ArgHandler
from src.manager.util import get_kernel_path

DEBUG = 0

class CudaKernelRunner:
    def __init__(self, spec, config_space):
        self.spec = spec
        self.arg_handler = ArgHandler(self.spec)
        self.config_space = config_space if config_space else ConfigSpace(self.spec["ConfigurationSpace"])
        self.kernel_spec = self.spec["KernelSpecification"]
        self.original_compiler_options = copy.deepcopy(self.kernel_spec["CompilerOptions"])
        self.results = []
        self.stream = cp.cuda.Stream()
        self.timers = {
                "runtime_start": cp.cuda.Event(),
                "runtime_end": cp.cuda.Event(),
                "total_start": cp.cuda.Event(),
                "total_end": cp.cuda.Event(),
                "start": cp.cuda.Event(),
                "end": cp.cuda.Event(),
        }
        self.dev = cp.cuda.Device(0)
        self.result = None
        self.tuning_config = {}
        self.context = {}

    def run(self, tuning_config, result):
        self.reset_context()
        self.tuning_config = tuning_config
        result.config = tuning_config

        self.add_to_context(self.spec["BenchmarkConfig"])
        self.add_to_context(self.tuning_config)
        result = self.run_kernel(self.get_launch_config(), tuning_config, result)
        self.reset_context()
        return result

    def generate_compiler_options(self, tuning_config):
        benchmark_config = self.spec.get("BenchmarkConfig", {})
        compiler_options = self.kernel_spec.get("CompilerOptions", [])
        for (key, val) in tuning_config.items():
            compiler_options.append(f"-D{key}={val}")
        for (key, val) in benchmark_config.items():
            compiler_options.append(f"-D{key}={val}")
        for (key, val) in self.result.launch.items():
            compiler_options.append(f"-D{key}={val}")
        return compiler_options

    def reset_context(self):
        self.context = {}
        self.kernel_spec["CompilerOptions"] = copy.deepcopy(self.original_compiler_options)
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def add_to_context(self, d):
        self.context.update(d)

    def get_context(self):
        return self.context

    def get_launch_config(self):
        kernel_spec = self.spec["KernelSpecification"]

        launch_config = {
            "GRID_SIZE_X": eval(str(kernel_spec["GlobalSize"].get("X", 1)), self.get_context()),
            "GRID_SIZE_Y": eval(str(kernel_spec["GlobalSize"].get("Y", 1)), self.get_context()),
            "GRID_SIZE_Z": eval(str(kernel_spec["GlobalSize"].get("Z", 1)), self.get_context()),
            "BLOCK_SIZE_X": eval(str(kernel_spec["LocalSize"].get("X", 1)), self.get_context()),
            "BLOCK_SIZE_Y": eval(str(kernel_spec["LocalSize"].get("Y", 1)), self.get_context()),
            "BLOCK_SIZE_Z": eval(str(kernel_spec["LocalSize"].get("Z", 1)), self.get_context()),
        }

        global_size_type = kernel_spec.get("GlobalSizeType", "CUDA")
        if global_size_type.lower() == "opencl":
            launch_config["GRID_SIZE_X"] = math.ceil(launch_config["GRID_SIZE_X"]/launch_config["BLOCK_SIZE_X"])
            launch_config["GRID_SIZE_Y"] = math.ceil(launch_config["GRID_SIZE_Y"]/launch_config["BLOCK_SIZE_Y"])
            launch_config["GRID_SIZE_Z"] = math.ceil(launch_config["GRID_SIZE_Z"]/launch_config["BLOCK_SIZE_Z"])

        self.add_to_context(launch_config)

        if kernel_spec.get("SharedMemory"):
            launch_config["SHARED_MEMORY_SIZE"] = eval(str(kernel_spec["SharedMemory"]), self.get_context())

        self.grid_dim = (launch_config["GRID_SIZE_X"], launch_config["GRID_SIZE_Y"], launch_config["GRID_SIZE_Z"])
        self.block_dim = (launch_config["BLOCK_SIZE_X"], launch_config["BLOCK_SIZE_Y"], launch_config["BLOCK_SIZE_Z"])
        self.shared_mem_size = launch_config.get("SHARED_MEMORY_SIZE", 0)
        return launch_config

    def compile_kernel(self, tuning_config):
        jitify = self.spec["BenchmarkConfig"].get("jitify", False)
        if jitify == False and self.kernel_spec.get("Misc", False):
            jitify = self.kernel_spec["Misc"].get("jitify", False)
        with open(get_kernel_path(self.spec), 'r') as f:
            module = cp.RawModule(code=f.read(),
                    backend=self.spec["BenchmarkConfig"].get("backend", "nvrtc"),
                    options=tuple(self.generate_compiler_options(tuning_config)),
                    jitify=jitify)

        self.start_timer()
        kernel = module.get_function(self.kernel_spec["KernelName"])
        self.kernel = kernel
        self.result.compile_time = self.get_duration()

    def start_timer(self, name="start"):
        self.timers[name].record(stream=self.stream)

    def get_duration(self, start_name="start", end_name="end"):
        self.timers[end_name].record(stream=self.stream)
        self.dev.synchronize()
        return cp.cuda.get_elapsed_time(self.timers[start_name], self.timers[end_name]) / 1000

    def launch_kernel(self, args_tuple, launch_config):
        self.start_timer("runtime_start")
        self.result.runtimes = []
        for _ in range(launch_config.get("iterations", 10)):
            self.start_timer()
            self.kernel(grid=self.grid_dim, block=self.block_dim, args=args_tuple, shared_mem=self.shared_mem_size)
            self.result.runtimes.append(self.get_duration())

        self.result.objective = self.get_duration("runtime_start", "runtime_end")

    def get_kernel_string(self) -> str:
        """ Reads in the kernel as a string """
        kernel_string = ""
        with open(get_kernel_path(self.spec), 'r') as f:
            kernel_string += f.read()
        return kernel_string

    def invalid_result(self, msg, error=None):
        self.result.validity = msg
        self.result.correctness = 0
        self.result.runtimes = []
        if error:
            self.result.error = error
        return self.result

    def run_kernel(self, launch_config, tuning_config, result):
        self.result = result
        self.result.launch = launch_config

        if tuple(tuning_config.values()) not in self.config_space:
            return self.invalid_result("Config invalid")
        try:
            self.compile_kernel(tuning_config)
        except Exception as e:
            print(e)
            return self.invalid_result("Compile exception", e)

        t0 = time.time()
        args_tuple, cmem_args = tuple(self.arg_handler.populate_args(self.kernel_spec["Arguments"]))
        self.result.arg_time = time.time() - t0

        try:
            self.launch_kernel(args_tuple, launch_config)
        except Exception as e:
            return self.invalid_result("Launch exception", e)

        if DEBUG:
            correctness = correctness_funcs[self.kernel_spec["KernelName"]]
            correctness(tuple(args), args_tuple, tuning_config, launch_config)
        return self.result
