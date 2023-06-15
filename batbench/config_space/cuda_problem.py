from typing import Dict, Any, Optional

from batbench.backends.kernelbackend.kerneltuner_runner import KernelBackend
from batbench.backends.cuda_kernel_runner.arg_handler import ArgHandler
from batbench.result.result import Result

from .config_space import ConfigSpace
from .problem import Problem



class CUDAProgram:
    """
    A class representing a CUDA program.

    Attributes:
        kernel_name (str): The name of the kernel function.
    """
    def __init__(self, kernel_name: str) -> None:
        self._kernel_name = kernel_name


class CUDAProblem(Problem):
    """
    A class representing a CUDA problem.

    Inherits from:
        Problem: A class representing a problem.

    Attributes:
        program (CUDAProgram): The CUDA program associated with this problem.
        language (str): The programming language used for this problem.
        config_space (ConfigSpace): The configuration space for this problem.
        function_args (Any): The arguments for the CUDA kernel function.
    """
    def __init__(self, kernel_name: str, spec: Optional[Dict[str, Any]] = None,
                 run_settings: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._kernel_name = kernel_name
        self._program = CUDAProgram(kernel_name)
        self._language = "CUDA"
        self.cuda_backend = 'cupy'
        self.function_args = None

        self.spec = spec if spec is not None else {}
        self.spec["BenchmarkConfig"] = { "iterations": 10 }
        if "ConfigurationSpace" in self.spec:
            self._config_space = ConfigSpace(self.spec["ConfigurationSpace"])
            self.function_args = ArgHandler(self.spec).populate_args()
            self.runner = KernelBackend(self.spec, self.config_space, self.function_args)

        self.run_settings = run_settings if run_settings is not None else {}

    @property
    def program(self) -> CUDAProgram:
        """
        Returns the CUDA program associated with this problem.

        Returns:
            CUDAProgram: The CUDA program associated with this problem.
        """
        return self._program

    @property
    def language(self) -> str:
        """
        Returns the programming language used for this problem.

        Returns:
            str: The programming language used for this problem.
        """
        return self._language

    @property
    def config_space(self) -> ConfigSpace:
        """
        Returns the configuration space for this problem.

        Returns:
            ConfigSpace: The configuration space for this problem.
        """
        return self._config_space

    @config_space.setter
    def config_space(self, value: ConfigSpace) -> None:
        """
        Sets the configuration space for this problem.

        Args:
            value (ConfigSpace): The configuration space to set for this problem.
        """
        self._config_space = value

    def get_args(self) -> Any:
        """
        Returns the arguments for the CUDA kernel function.

        Returns:
            Any: The arguments for the CUDA kernel function.
        """
        return self.function_args

    def run(self, tuning_config: Dict[str, Any], result: Result) -> Result:
        """
        Runs the CUDA kernel function with the given tuning configuration and returns the result.

        Args:
            tuning_config (Dict[str, Any]): The tuning configuration to use for the kernel function.
            result (Result): The result object to store the results of the kernel function.

        Returns:
            Result: The result object containing the results of the kernel function.
        """
        return self.runner.run(tuning_config, result)
