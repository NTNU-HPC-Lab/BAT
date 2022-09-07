import json

from src.readers.python.config_space import ConfigSpace


class Manager:
    def __init__(self, args):
        self.spec = self.get_spec(args.json)
        self.config_space = ConfigSpace(self.spec["configurationSpace"])
        lang = self.spec["kernelSpecification"]["language"]
        if lang == "CUDA":
            from src.readers.python.cuda.cuda_kernel_runner import CudaKernelRunner
            from src.readers.python.cuda.arg_handler import ArgHandler
            self.runner = CudaKernelRunner(self.spec, self.config_space)
            self.arg_handler = ArgHandler(self.spec)
        self.filename = "optuna-results.json"
        self.testing = 0
        self.results = []

    def write(self):
        dump_results = {"results": []}
        with open(self.filename, 'a') as f:
            for result in self.results:
                dump_results["results"].append(result.serialize())
            json.dump(dump_results, f, indent=4)

    def get_spec(self, json_path):
        with open(json_path, 'r') as f:
            r = json.load(f)
        return r

    def run(self, tuning_config, result):
        result = self.runner.run(tuning_config, result)
        self.add_result(result)
        return result

    def add_result(self, result):
        self.results.append(result)

    def get_last(self):
        return self.results[-1]

