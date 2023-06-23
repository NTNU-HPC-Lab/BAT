import argparse
import pytest
from batbench.__main__ import add_standard_arguments_to_parser
from batbench.manager.experiment_manager import ExperimentManager

pytestmark = pytest.mark.backends

def benchmark_setup(benchmark):
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)
    parser.add_argument('--gpu_name', type=str, default='A4000', help='The CUDA GPU to run on')
    parser.add_argument('--cache', type=str, default='', help='The cache to use')
    args = parser.parse_args()
    args.experiment_settings = "./tests/gpu/integration/experiment-settings-test.json"
    args.benchmarks = [benchmark]
    args.cleanup = True
    args.tuner = ["kerneltuner"]
    return args

def test_full_gemm_kerneltuner():
    ExperimentManager().start(benchmark_setup("GEMM"))

#def test_full_expdist_kerneltuner():
#    ExperimentManager().start(benchmark_setup("expdist"))

#def test_full_pnpoly_kerneltuner():
#    ExperimentManager().start(benchmark_setup("pnpoly"))

#def test_full_dedisp_kerneltuner():
#    ExperimentManager().start(benchmark_setup("dedisp"))

#def test_full_hotspot_kerneltuner():
#    ExperimentManager().start(benchmark_setup("hotspot"))

#def test_full_convolution_kerneltuner():
#    ExperimentManager().start(benchmark_setup("convolution"))

#def test_full_nbody_kerneltuner():
#    ExperimentManager().start(benchmark_setup("nbody"))
