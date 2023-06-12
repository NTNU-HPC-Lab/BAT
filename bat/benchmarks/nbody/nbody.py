from bat.backends.cuda_kernel_runner.cuda_kernel_runner import CudaKernelRunner
from bat.config_space.cuda_problem import CUDAProblem
from bat.util import get_spec_by_name

class Nbody(CUDAProblem):
    def __init__(self, run_settings) -> None:
        super().__init__("nbody", get_spec_by_name("nbody"))
        self.runner = CudaKernelRunner(self.spec, self.config_space)

