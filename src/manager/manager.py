import json
import os

from src.config_space import ConfigSpace
from src.result.dataset import Dataset
from src.manager.util import get_spec, get_search_spec


class ExperimentManager:
    def __init__(self):
        self.runner_dict = {
            "opentuner": self.run_opentuner,
            "optuna": self.run_optuna,
            "kerneltuner": self.run_kerneltuner,
            "smac": self.run_smac,
            "smac3": self.run_smac
        }
    @staticmethod
    def run_opentuner(args):
        from src.tuners.opentuner_runner import OpenTunerT
        print(OpenTunerT.main(args))

    @staticmethod
    def run_smac(args):
        from src.tuners.smac3_runner.smac3_runner import SMAC3
        print(SMAC3().main(args))

    @staticmethod
    def run_optuna(args):
        from src.tuners.optuna_runner import Optuna
        print(Optuna().main(args))

    @staticmethod
    def run_kerneltuner(args):
        from src.tuners.kerneltuner_runner import KernelTuner
        print(KernelTuner().main(args))

    def start(self, args):
        self.search_spec = get_search_spec()
        benchmarks = args.benchmarks if args.benchmarks else self.search_spec["BenchmarkConfig"]["Benchmarks"]

        if benchmarks[0].lower() == "all":
            benchmarks = ["GEMM","nbody", "MD5Hash", "FFT", "TRIAD"]

        tuner = args.tuner if args.tuner else self.search_spec["SearchSettings"]["TunerName"]

        for benchmark in benchmarks:
            args.json = "./benchmarks/{}/{}-CAFF.json".format(benchmark, benchmark)
            if tuner is not None:
                print("Running {} with {}".format(tuner, args))
                self.runner_dict[tuner](args)



class Manager:
    def __init__(self, args):
        self.root_results_path = "./results"
        self.spec = get_spec(args.json)
        self.search_spec = get_search_spec()
        self.validate_schema(self.spec)

        self.config_space = ConfigSpace(self.spec["ConfigurationSpace"])
        self.dataset = Dataset(args.json)
        self.search_settings = self.spec["SearchSettings"]
        lang = self.spec["KernelSpecification"]["Language"]
        if lang == "CUDA":
            from src.cuda_kernel_runner import CudaKernelRunner, ArgHandler
            self.runner = CudaKernelRunner(self.spec, self.config_space, self.search_spec)
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

