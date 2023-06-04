from src.config_space.config_space import ConfigSpace
from src.kernelbackend.kerneltuner_runner import KernelBackend
from src.cuda_kernel_runner.arg_handler import ArgHandler
from src.config_space.problem import Problem


class CUDAProgram:

    def __init__(self, kernel_name) -> None:
        self._kernel_name = kernel_name


class CUDAProblem(Problem):
    def __init__(self, kernel_name, spec=None, run_settings={}) -> None:
        super().__init__()
        self._kernel_name = kernel_name
        self._program = CUDAProgram(kernel_name)
        self._language = "CUDA"
        self.lang = 'cupy'
        if spec is not None:
            self.spec = spec
            self.spec["BenchmarkConfig"] = {}
            self.spec["BenchmarkConfig"]["iterations"] = 10
            if spec.get("ConfigurationSpace"):
                self.config_space = ConfigSpace(self.spec["ConfigurationSpace"])
                self.function_args = ArgHandler(self.spec).populate_args()
                self.runner = KernelBackend(self.spec, self.config_space, self.function_args)

    @property
    def program(self) -> CUDAProgram:
        return self._program

    @property
    def language(self):
        return self._language

    @property
    def config_space(self):
        return self._config_space

    @config_space.setter
    def config_space(self, value):
        self._config_space = value

    def get_args(self):
        return self.function_args

    def run(self, tuning_config, result):
        return self.runner.run(tuning_config, result)

