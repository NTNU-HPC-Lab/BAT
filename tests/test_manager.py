import pytest
import argparse
from main import add_standard_arguments_to_parser
from src.manager import Manager

def helper_setup(parser):
    args = parser.parse_args()
    args.json = "./benchmarks/TRIAD/TRIAD-CAFF.json"
    args.experiment_settings = "./experiment-settings.json"
    args.benchmark = "TRIAD"
    args.cleanup = True
    return args

def test_manager_init():
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)
    args = helper_setup(parser)
    manager = Manager(args)
    manager.validate_schema(manager.spec)
    manager.dataset.delete_files()

def test_full_mintuner():
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)
    args = helper_setup(parser)
    from src.tuners.mintuner_runner.mintuner_runner import MinTuner
    MinTuner().main(args)

def test_full_kerneltuner():
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)
    parser.add_argument('--gpu_name', type=str, default='A4000', help='The CUDA GPU to run on')
    args = helper_setup(parser)
    from src.tuners.kerneltuner_runner import KernelTuner
    KernelTuner().main(args)

def test_full_opentuner():
    import opentuner
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    parser = add_standard_arguments_to_parser(parser)
    args = helper_setup(parser)
    from src.tuners.opentuner_runner import OpenTunerT
    print(args)
    OpenTunerT.main(args)

def test_full_smac():
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)
    args = helper_setup(parser)
    from src.tuners.smac3_runner.smac3_runner import SMAC3
    SMAC3().main(args)
