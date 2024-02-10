from batbench.config_space.cuda_problem import CUDAProblem
from batbench.util import get_spec_by_name

class Nbody(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("nbody", get_spec_by_name("nbody"), runner="cuda_kernel_runner")
        #self.runner.run_reference(self.config_space.get_default_config())
