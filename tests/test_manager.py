import pytest
import argparse
from main import add_standard_arguments_to_parser
from src.manager import Manager

def test_manager_init():
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)
    args = parser.parse_args()
    args.json = "./benchmarks/TRIAD/TRIAD-CAFF.json"
    args.experiment_settings = "./experiment-settings.json"
    args.benchmark = "TRIAD"
    manager = Manager(args)
    manager.validate_schema(manager.spec)
    manager.dataset.delete_files()

def test_full_mintuner():
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)
    args = parser.parse_args()
    args.json = "./benchmarks/TRIAD/TRIAD-CAFF.json"
    args.experiment_settings = "./experiment-settings.json"
    args.benchmark = "TRIAD"
    from src.tuners.mintuner_runner.mintuner_runner import MinTuner
    MinTuner().main(args)
