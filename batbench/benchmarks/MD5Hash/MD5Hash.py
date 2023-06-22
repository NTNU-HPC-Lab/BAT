from batbench.config_space.cuda_problem import CUDAProblem
from batbench.util import get_spec_by_name

class MD5Hash(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("MD5Hash", get_spec_by_name("MD5Hash"))
        self.runner.run_reference(self.config_space.get_default_config())
