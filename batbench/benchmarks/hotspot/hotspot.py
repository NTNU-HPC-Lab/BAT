from batbench.config_space.cuda_problem import CUDAProblem
from batbench.util import get_spec_by_name

class Hotspot(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("hotspot", get_spec_by_name("hotspot"), cuda_backend="CUDA")
        self.runner.run_reference(self.config_space.get_default_config())
