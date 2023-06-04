from src.config_space.cuda_problem import CUDAProblem
from src.manager.util import get_spec_by_name

class TRIAD(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("TRIAD", get_spec_by_name("TRIAD"))    

