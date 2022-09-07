import time
import cupy as cp

from src.result import Result
from src.config_space import ConfigSpace

from .arg_handler import ArgHandler
from src.manager import get_kernel_path


DEBUG = 0


class CudaKernelRunner:
    def __init__(self, spec, config_space=None):
        self.spec = spec
        self.arg_handler = ArgHandler(self.spec)
        self.config_space = config_space if config_space else ConfigSpace(self.spec["configurationSpace"])
        self.kernel_spec = self.spec["kernelSpecification"]
        self.results = []
        self.result = Result(self.spec)

    def run(self, tuning_config, result):
        result.config = tuning_config
        return self.run_kernel(self.get_launch_config(tuning_config), tuning_config, result)

    def generate_compiler_options(self, tuning_config):
        benchmark_config = self.spec["benchmarkConfig"]
        compiler_options = self.kernel_spec["compilerOptions"]
        for (key, val) in tuning_config.items():
            compiler_options.append("-D{}={}".format(key, val))
        for (key, val) in benchmark_config.items():
            compiler_options.append("-D{}={}".format(key, val))
        return compiler_options

    def get_launch_config(self, tuning_config):
        kernel_spec = self.spec["kernelSpecification"]
        benchmark_config = self.spec["benchmarkConfig"]
        for name, value in benchmark_config.items():
            locals()[name] = eval(str(value))
        for name, value in tuning_config.items():
            locals()[name] = value

        launch_config = {
            "GRID_SIZE_X": eval(str(kernel_spec["gridSize"]["X"])),
            "GRID_SIZE_Y": eval(str(kernel_spec["gridSize"]["Y"])),
            "GRID_SIZE_Z": eval(str(kernel_spec["gridSize"]["Z"])),
            "BLOCK_SIZE_X": eval(str(kernel_spec["blockSize"]["X"])),
            "BLOCK_SIZE_Y": eval(str(kernel_spec["blockSize"]["Y"])),
            "BLOCK_SIZE_Z": eval(str(kernel_spec["blockSize"]["Z"])),
        }
        for name, value in launch_config.items():
            locals()[name] = value

        if kernel_spec.get("sharedMemory"):
            launch_config["SHARED_MEMORY_SIZE"] = eval(str(kernel_spec["sharedMemory"]))

        self.grid_dim = (launch_config["GRID_SIZE_X"], launch_config["GRID_SIZE_Y"], launch_config["GRID_SIZE_Z"])
        self.block_dim = (launch_config["BLOCK_SIZE_X"], launch_config["BLOCK_SIZE_Y"], launch_config["BLOCK_SIZE_Z"])
        self.shared_mem_size = launch_config.get("SHARED_MEMORY_SIZE", 0)
        return launch_config

    def compile_kernel(self, tuning_config):
        with open(get_kernel_path(self.spec), 'r') as f:
            module = cp.RawModule(code=f.read(),
                    backend=self.spec["benchmarkConfig"].get("backend", "nvrtc"),
                    options=tuple(self.generate_compiler_options(tuning_config)),
                    jitify=self.spec["benchmarkConfig"].get("jitify", False))
        t0 = time.time()
        self.kernel = module.get_function(self.kernel_spec["kernelName"])

        self.result.compile_time = time.time() - t0

    def launch_kernel(self, args_tuple, launch_config):
        t0 = time.time()
        self.result.runtimes = []
        for i in range(launch_config.get("ITERATIONS", 10)):
            t00 = time.time()
            self.kernel(grid=self.grid_dim, block=self.block_dim, args=args_tuple, shared_mem=self.shared_mem_size)
            inner_duration = time.time() - t00
            self.result.runtimes.append(inner_duration)

        self.result.objective = time.time() - t0

    def get_kernel_string(self) -> str:
        """ Reads in the kernel as a string """
        kernel_string = ""
        with open(get_kernel_path(self.spec), 'r') as f:
            kernel_string += f.read()
        return kernel_string

    def invalid_result(self, msg):
        self.result.validity = msg
        self.result.correctness = 0
        self.result.objective = float('inf')
        self.result.runtimes = []
        return self.result

    def run_kernel(self, launch_config, tuning_config, result):
        self.result = result
        self.result.launch = launch_config

        if tuple(tuning_config.values()) not in self.config_space:
            return self.invalid_result("Config invalid")
        try:
            self.compile_kernel(tuning_config)
        except Exception as e:
            return self.invalid_result("Compile exception")

        args_tuple = tuple(self.arg_handler.populate_args(self.kernel_spec["arguments"]))

        try:
            self.launch_kernel(args_tuple, launch_config)
        except Exception as e:
            return self.invalid_result("Launch exception")

        if DEBUG:
            correctness = correctness_funcs[self.kernel_spec["kernelName"]]
            correctness(tuple(args), args_tuple, tuning_config, launch_config)
        return self.result
