import json
import os
import datetime
import logging
import jsonschema

from batbench import benchmarks
from batbench.result.dataset import Dataset
from batbench.result.zenodo import Zenodo
from batbench.util import get_spec, SCHEMA_PATH


# Create a custom logger
log = logging.getLogger(__name__)


def get_budget_trials(spec):
    budgets = spec.get("Budget", [])
    for budget in budgets:
        if budget["Type"] == "ConfigurationCount":
            return budget["BudgetValue"]
    raise ValueError("No budget value found")


benchmark_map = {
    "dedisp": benchmarks.Dedispersion,
    "expdist": benchmarks.Expdist,
    "GEMM": benchmarks.GEMM,
    "hotspot": benchmarks.Hotspot,
    "pnpoly": benchmarks.PnPoly,
    "convolution": benchmarks.Convolution,
    "nbody": benchmarks.Nbody,
    "TRIAD": benchmarks.TRIAD,
    "MD5Hash": benchmarks.MD5Hash,
}


class Manager:
    """
    The Manager class is responsible for managing the benchmarking process. 
    It initializes the benchmark problem, runs the tuning process, and manages the results. 
    It also handles the validation of the tuning configuration schema and 
    the uploading of the results to Zenodo.
    """
    def __init__(self, args):
        self.cleanup = args.cleanup
        self.root_results_path = "./results"

        experiment_settings = get_spec(args.experiment_settings)

        self.problem = benchmark_map[args.benchmark](experiment_settings)
        self.config_space = self.problem.config_space
        self.budget_trials = experiment_settings["Budget"][0]["BudgetValue"]
        print(f"General: {self.problem.spec['General']}")
        self.objective = self.problem.spec['General']['Objective'], 
        self.minimize = self.problem.spec['General']['Minimize']
        self.dataset = Dataset(experiment_settings, args.benchmark, self.objective, self.minimize)
        self.trial = 0
        self.total_time = 0

        self.testing = 0
        self.timestamp = datetime.datetime.now()
        self.result_timestamp = datetime.datetime.now()


    def validate_schema(self, spec):
        with open(f'{SCHEMA_PATH}/TuningSchema.json', 'r', encoding='utf-8') as file:
            schema = json.load(file)
        jsonschema.validate(instance=spec, schema=schema)

    @staticmethod
    def upload(root_results_path):
        datasets = os.listdir(root_results_path)
        Zenodo(datasets).upload()

    def finished(self):
        self.cleanup = False
        if self.cleanup:
            self.dataset.delete_files()

    def write(self):
        self.dataset.write_data()

    def run(self, tuning_config, result):
        dur = (datetime.datetime.now() - self.timestamp).total_seconds()
        #if dur > 600.0 or self.trial == self.budget_trials:
        if dur > 60000.0 or self.trial == self.budget_trials:
            print(dur, self.trial, self.budget_trials)
            raise KeyboardInterrupt
        if list(tuning_config.values()) not in self.problem.config_space:
            result.validity = "KnownConstraintsViolated"
            result.objective = 10000 if self.minimize else 0
            result.correctness = 0.0
        else:
            result = self.problem.run(tuning_config, result)
            result.total_time = (datetime.datetime.now() - self.result_timestamp).total_seconds()
            self.trial += 1
            self.total_time += result.total_time
            estimated_time = (self.budget_trials/self.trial) * self.total_time
            print(f"Trials: {self.trial}/{self.budget_trials} | Total time: {self.total_time:.0f}s | Estimated Time: {estimated_time:.0f}s",
                  end="\r")
            self.result_timestamp = datetime.datetime.now()

        self.dataset.add_result(result)
        return result
