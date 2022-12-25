import json
import os

from src.config_space import ConfigSpace
from src.result.dataset import Dataset
from src.manager.util import get_spec, write_spec


class ExperimentManager:
    def __init__(self):
        self.runner_dict = {
            "opentuner": self.run_opentuner,
            "optuna": self.run_optuna,
            "kerneltuner": self.run_kerneltuner,
            "smac": self.run_smac,
            "smac3": self.run_smac,
            "mintuner": self.run_mintuner,
            "ktt": self.run_ktt
        }
    @staticmethod
    def run_opentuner(args):
        from src.tuners.opentuner_runner import OpenTunerT
        print(OpenTunerT.main(args))

    @staticmethod
    def run_mintuner(args):
        from src.tuners.mintuner_runner.mintuner_runner import MinTuner
        print(MinTuner().main(args))

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

    @staticmethod
    def run_ktt(args):
        from src.tuners.KTT_runner import KTTRunner
        print(KTTRunner().main(args))


    @staticmethod
    def update_experiment_settings(args):
        experiment_settings = get_spec(args.experiment_settings)
        if args.tuner:
            experiment_settings["SearchSettings"]["TunerName"] = args.tuner
        if args.trials:
            experiment_settings["Budget"]["BudgetValue"] = args.trials
        if args.logging:
            experiment_settings["General"]["LoggingLevel"] = args.logging
        if args.output_format:
            experiment_settings["General"]["OutputFormat"] = args.output_format
        if args.benchmarks:
            if args.benchmarks[0].lower() == "all":
                benchmarks = ["GEMM","nbody", "MD5Hash", "FFT", "TRIAD"]
            else:
                benchmarks = args.benchmarks
            experiment_settings["BenchmarkConfig"]["Benchmarks"] = benchmarks
        write_spec(experiment_settings, args.experiment_settings)
        return experiment_settings

    def start(self, args):
        experiment_settings = self.update_experiment_settings(args)
        tuners = experiment_settings["SearchSettings"]["TunerName"]
        benchmarks = experiment_settings["BenchmarkConfig"]["Benchmarks"]

        if args.json:
            for tuner in tuners:
                print(f"Running {tuner} with {args}")
                self.runner_dict[tuner.lower()](args)
            return

        for benchmark in benchmarks:
            args.json = f"./benchmarks/{benchmark}/{benchmark}-CAFF.json"
            for tuner in tuners:
                print(f"Running {tuner} with {args}")
                self.runner_dict[tuner.lower()](args)



class Manager:
    def __init__(self, args):
        self.root_results_path = "./results"
        self.spec = get_spec(args.json)
        self.experiment_settings = get_spec(args.experiment_settings)
        self.validate_schema(self.spec)
        self.budget = self.experiment_settings["Budget"]["BudgetValue"]
        self.trial = 0
        self.total_time = 0

        self.config_space = ConfigSpace(self.spec["ConfigurationSpace"])
        self.dataset = Dataset(args.json)
        lang = self.spec["KernelSpecification"]["Language"]
        if lang == "CUDA":
            from src.cuda_kernel_runner import CudaKernelRunner, ArgHandler
            self.runner = CudaKernelRunner(self.spec, self.config_space, self.experiment_settings)
            self.arg_handler = ArgHandler(self.spec)
        else:
            from src.simulated_runner import SimulatedRunner, ArgHandler
            self.runner = SimulatedRunner(self.spec, self.config_space, self.experiment_settings)
            self.arg_handler = ArgHandler(self.spec)

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
        self.trial += 1
        self.total_time += result.total_time
        estimated_time = (self.budget/self.trial) * self.total_time

        print(f"Trials: {self.trial}/{self.budget} | Total time: {self.total_time:.0f}s | Estimated Time: {estimated_time:.0f}s", end="\r")
        self.dataset.add_result(result)
        return result

    def get_last(self):
        return self.results[-1]

