import json
import os
import datetime
import jsonschema
import logging
from benchmarks.MD5Hash.MD5Hash import MD5Hash
from benchmarks.TRIAD.TRIAD import TRIAD
from benchmarks.convolution.convolution import Convolution
from benchmarks.nbody.nbody import Nbody
from benchmarks.pnpoly.pnpoly import PnPoly

from src.result.dataset import Dataset
from src.result.zenodo import Zenodo
from src.manager.util import get_spec

from benchmarks.dedisp.dedispersion import Dedispersion
from benchmarks.expdist.expdist import Expdist
from benchmarks.GEMM.gemm import GEMM
from benchmarks.hotspot.hotspot import Hotspot

# Create a custom logger
log = logging.getLogger(__name__)


def get_budget_trials(spec):
    budgets = spec.get("Budget", [])
    for budget in budgets:
        if budget["Type"] == "ConfigurationCount":
            return budget["BudgetValue"]


benchmark_map = {
    "dedisp": Dedispersion,
    "expdist": Expdist,
    "GEMM": GEMM,
    "hotspot": Hotspot,
    "pnpoly": PnPoly,
    "convolution": Convolution,
    "nbody": Nbody,
    "TRIAD": TRIAD,
    "MD5Hash": MD5Hash,
}


class Manager:
    def __init__(self, args):
        self.cleanup = args.cleanup
        self.root_results_path = "./results"

        experiment_settings = get_spec(args.experiment_settings)

        self.problem = benchmark_map[args.benchmark](experiment_settings)
        self.config_space = self.problem.config_space
        experiment_settings = experiment_settings
        self.budget_trials = experiment_settings["Budget"][0]["BudgetValue"]
        self.dataset = Dataset(experiment_settings, args.benchmark)
        self.trial = 0
        self.total_time = 0

        self.testing = 0
        self.timestamp = datetime.datetime.now()
        self.result_timestamp = datetime.datetime.now()


    def validate_schema(self, spec):
        with open('schemas/TuningSchema/TuningSchema.json', 'r') as f:
            schema = json.load(f)
        jsonschema.validate(instance=spec, schema=schema)

    @staticmethod
    def upload(root_results_path):
        datasets = os.listdir(root_results_path)
        z = Zenodo(datasets)
        z.upload()

    def finished(self):
        if self.cleanup:
            self.dataset.delete_files()

    def write(self):
        self.dataset.write_data()

    def run(self, tuning_config, result):
        dur = (datetime.datetime.now() - self.timestamp).total_seconds()
        if dur > 600.0 or self.trial == self.budget_trials:
            raise KeyboardInterrupt
        if list(tuning_config.values()) not in self.problem.config_space:
            result.validity = "KnownConstraintsViolated"
        else:
            result = self.problem.run(tuning_config, result)
            result.total_time = (datetime.datetime.now() - self.result_timestamp).total_seconds()
            self.trial += 1
            self.total_time += result.total_time
            estimated_time = (self.budget_trials/self.trial) * self.total_time

            print(f"Trials: {self.trial}/{self.budget_trials} | Total time: {self.total_time:.0f}s | Estimated Time: {estimated_time:.0f}s", end="\r")
            self.result_timestamp = datetime.datetime.now()

        self.dataset.add_result(result)
        return result

    def get_last(self):
        return self.dataset.get_last()

