import json
import os
import datetime
import jsonschema
import logging

from src.config_space import ConfigSpace
from src.result.dataset import Dataset
from src.manager.util import get_spec, write_spec

from src.cuda_kernel_runner import ArgHandler
from src.kernelbackend import KernelBackend
from src.result.zenodo import Zenodo

# Create a custom logger
log = logging.getLogger(__name__)


def get_budget_trials(spec):
    budgets = spec.get("Budget", [])
    for budget in budgets:
        if budget["Type"] == "ConfigurationCount":
            return budget["BudgetValue"]


class Manager:
    def __init__(self, args):
        self.cleanup = args.cleanup
        self.root_results_path = "./results"
        self.spec = get_spec(args.json)
        self.spec.update(get_spec(args.experiment_settings))
        self.spec["General"]["BenchmarkName"] = args.benchmark
        self.validate_schema(self.spec)
        # write_spec(self.spec, args.json)
        self.budget_trials = get_budget_trials(self.spec)
        self.trial = 0
        self.total_time = 0

        self.config_space = ConfigSpace(self.spec["ConfigurationSpace"])
        self.dataset = Dataset(self.spec)
        lang = self.spec["KernelSpecification"]["Language"]
        if lang == "CUDA":
            self.arg_handler = ArgHandler(self.spec)

            #self.runner = CudaKernelRunner(self.spec, self.config_space)
            self.runner = KernelBackend(args, self)
        else:
            raise NotImplementedError(f"Language {lang} not supported")
            #self.arg_handler = ArgHandler(self.spec)
            #self.runner = SimulatedRunner(self.spec, self.config_space)

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
        if list(tuning_config.values()) not in self.config_space:
            result.validity = "KnownConstraintsViolated"
        else:    
            result = self.runner.run(tuning_config, result)
            result.total_time = (datetime.datetime.now() - self.result_timestamp).total_seconds()
            self.trial += 1
            self.total_time += result.total_time
            estimated_time = (self.budget_trials/self.trial) * self.total_time

            print(f"Trials: {self.trial}/{self.budget_trials} | Total time: {self.total_time:.0f}s | Estimated Time: {estimated_time:.0f}s", end="\r")
            self.result_timestamp = datetime.datetime.now()
        
        self.dataset.add_result(result)
        return result

    def get_last(self):
        return self.results[-1]

