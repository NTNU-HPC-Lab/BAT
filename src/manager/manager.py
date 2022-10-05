import json

from src.config_space import ConfigSpace
from src.result.dataset import Dataset

class Manager:
    def __init__(self, args):
        self.spec = self.get_spec(args.json)
        self.validate_schema()
        self.config_space = ConfigSpace(self.spec["ConfigurationSpace"])
        self.dataset = Dataset(self.spec)
        self.search_settings = self.spec["SearchSettings"]
        lang = self.spec["KernelSpecification"]["Language"]
        if lang == "CUDA":
            from src.cuda_kernel_runner import CudaKernelRunner, ArgHandler
            self.runner = CudaKernelRunner(self.spec, self.config_space)
            self.arg_handler = ArgHandler(self.spec)


        #self.filename = "optuna-results.hdf5"
        self.testing = 0

    def validate_schema(self):
        from jsonschema import validate
        with open('schemas/TuningSchema/TuningSchema.json', 'r') as f:
            schema = json.loads(f.read())
        validate(instance=self.spec, schema=schema)


    def write(self):
        self.dataset.write_data()
        self.dataset.write_metadata()

    def get_spec(self, json_path):
        with open(json_path, 'r') as f:
            r = json.load(f)
        return r

    def run(self, tuning_config, result):
        result = self.runner.run(tuning_config, result)
        result.calculate_time()
        self.dataset.add_result(result)
        return result

    def get_last(self):
        return self.results[-1]

