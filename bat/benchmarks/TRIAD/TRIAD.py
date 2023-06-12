from bat.config_space.cuda_problem import CUDAProblem
from bat.util import get_spec_by_name

class TRIAD(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("TRIAD", get_spec_by_name("TRIAD"))    

