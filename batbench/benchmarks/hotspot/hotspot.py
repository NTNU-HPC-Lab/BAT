from batbench.config_space.cuda_problem import CUDAProblem
from batbench.util import get_spec_by_name

class Hotspot(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("hotspot", get_spec_by_name("hotspot"))    

