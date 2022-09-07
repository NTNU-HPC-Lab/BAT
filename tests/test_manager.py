import pytest
from src.manager import Manager

def test_manager_init():
    args = {}
    args.json = "./src/benchmarks/TRIAD/TRIAD-CAFF.json"
    manager = Manager(args)
