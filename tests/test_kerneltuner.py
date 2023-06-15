import argparse
from batbench.__main__ import add_standard_arguments_to_parser
from batbench.manager.experiment_manager import ExperimentManager
from batbench.tuners.kerneltuner_runner import KernelTuner

def benchmark_setup(benchmark):
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)
    parser.add_argument('--gpu_name', type=str, default='A4000', help='The CUDA GPU to run on')
    parser.add_argument('--cache', type=str, default='', help='The cache to use')
    args = parser.parse_args()
    args.experiment_settings = "./experiment-settings.json"
    args.benchmarks = [benchmark]
    args.cleanup = True
    args.tuner = ["kerneltuner"]
    return args

def test_full_gemm_kerneltuner():
    ExperimentManager().start(benchmark_setup("GEMM"))

def test_full_pnpoly_kerneltuner():
    ExperimentManager().start(benchmark_setup("pnpoly"))

def test_full_hotspot_kerneltuner():
    ExperimentManager().start(benchmark_setup("hotspot"))

def test_full_convolution_kerneltuner():
    args = benchmark_setup("convolution")
    KernelTuner().main(args)

def test_full_nbody_kerneltuner():
    args = benchmark_setup("nbody")
    KernelTuner().main(args)

