from collections import OrderedDict

from batbench.config_space.cuda_problem import CUDAProblem
from batbench.util import get_spec_by_name

class GEMM(CUDAProblem):
    def __init__(self, run_settings) -> None:
        metrics = OrderedDict()
        spec = get_spec_by_name("GEMM")
        matrix_bytes = 1
        for i in range(0, 3):
            matrix_bytes *= spec["KernelSpecification"]["Arguments"][i]["FillValue"]

        metrics["GFLOPs"] = lambda p : (matrix_bytes/1e9) / (p["time"] / 1e3)

        print(run_settings)
        super().__init__("GEMM", spec, metrics=metrics, cuda_backend="HIP")
        self.runner.run_reference(self.config_space.default_config)
