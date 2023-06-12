from bat.config_space.cuda_problem import CUDAProblem
from bat.util import get_spec_by_name

class GEMM(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("GEMM", get_spec_by_name("GEMM"))    

