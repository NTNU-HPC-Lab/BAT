import json
import os

from src.config_space import ConfigSpace
from src.result.dataset import Dataset
from src.manager.util import get_spec

class Manager:
    def __init__(self, args):
        self.root_results_path = "./results"
        self.spec = get_spec(args.json)
        self.validate_schema(self.spec)

        self.config_space = ConfigSpace(self.spec["ConfigurationSpace"])
        self.dataset = Dataset(args.json)
        self.search_settings = self.spec["SearchSettings"]
        lang = self.spec["KernelSpecification"]["Language"]
        if lang == "CUDA":
            from src.cuda_kernel_runner import CudaKernelRunner, ArgHandler
            self.runner = CudaKernelRunner(self.spec, self.config_space)
            self.arg_handler = ArgHandler(self.spec)


        #self.filename = "optuna-results.hdf5"
        self.testing = 0

    def validate_schema(self, spec):
        from jsonschema import validate
        with open('schemas/TuningSchema/TuningSchema.json', 'r') as f:
            schema = json.loads(f.read())
        validate(instance=spec, schema=schema)

    @staticmethod
    def upload(root_results_path):
        from src.result.zenodo import Zenodo
        datasets = os.listdir(root_results_path)
        z = Zenodo(datasets)
        z.upload()

    def write(self):
        self.dataset.write_data()

    def run(self, tuning_config, result):
        result = self.runner.run(tuning_config, result)
        result.calculate_time()
        self.dataset.add_result(result)
        return result

    def get_last(self):
        return self.results[-1]

