from bat.config_space.cuda_problem import CUDAProblem
from bat.util import get_spec_by_name

class MD5Hash(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("MD5Hash", get_spec_by_name("MD5Hash"))    

