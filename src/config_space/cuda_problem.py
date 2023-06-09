from typing import Dict, Any

from src.config_space.config_space import ConfigSpace
from src.kernelbackend.kerneltuner_runner import KernelBackend
from src.cuda_kernel_runner.arg_handler import ArgHandler
from src.config_space.problem import Problem
from src.result.result import Result


class CUDAProgram:

    def __init__(self, kernel_name: str) -> None:
        self._kernel_name = kernel_name


class CUDAProblem(Problem):
    def __init__(self, kernel_name: str, spec: Dict[str, Any] = None, run_settings: Dict[str, Any] = {}) -> None:
        super().__init__()
        self._kernel_name = kernel_name
        self._program = CUDAProgram(kernel_name)
        self._language = "CUDA"
        self.lang = 'cupy'
        self._config_space = None
        self.function_args = None
        self.runner = None

        if spec is not None:
            self.spec = spec
            self.spec["BenchmarkConfig"] = { "iterations": 10 }
            if "ConfigurationSpace" in spec:
                self.config_space = ConfigSpace(self.spec["ConfigurationSpace"])
                self.function_args = ArgHandler(self.spec).populate_args()
                self.runner = KernelBackend(self.spec, self.config_space, self.function_args)

    @property
    def program(self) -> CUDAProgram:
        return self._program

    @property
    def language(self) -> str:
        return self._language

    @property
    def config_space(self) -> ConfigSpace:
        return self._config_space

    @config_space.setter
    def config_space(self, value: ConfigSpace) -> None:
        self._config_space = value

    def get_args(self) -> Any:
        return self.function_args

    def run(self, tuning_config: Dict[str, Any], result: Result) -> Result:
        return self.runner.run(tuning_config, result)

