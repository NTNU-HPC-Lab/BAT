import pytest
import argparse
from main import add_standard_arguments_to_parser
from src.manager import Manager

def test_manager_init():
    args = {}
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)
    args = parser.parse_args()
    args.json = "./benchmarks/TRIAD/TRIAD-CAFF.json"
    args.search_path = "./benchmarking-settings.json"
    manager = Manager(args)
