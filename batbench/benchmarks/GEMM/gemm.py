from batbench.config_space.cuda_problem import CUDAProblem
from batbench.util import get_spec_by_name

class GEMM(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("GEMM", get_spec_by_name("GEMM"))
        self.runner.run_reference(self.config_space.default_config)
