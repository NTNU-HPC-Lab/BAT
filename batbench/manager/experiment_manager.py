import logging
import os

from ..util import get_spec, write_spec

import batbench.tuners as tuners

# Create a custom logger
log = logging.getLogger(__name__)

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
        try:
            print(tuners.OpenTunerT.main(args))
        except NotImplementedError:
            log.error("Terminated due to NotImplementedError in OpenTunerT.main")
            return None

    @staticmethod
    def run_mintuner(args):
        print(tuners.MinTuner().main(args))

    @staticmethod
    def run_smac(args):
        print(tuners.SMAC3().main(args))

    @staticmethod
    def run_optuna(args):
        print(tuners.Optuna().main(args))

    @staticmethod
    def run_kerneltuner(args):
        print(tuners.KernelTuner().main(args))

    @staticmethod
    def run_ktt(args):
        print(tuners.KTTRunner().main(args))


    @staticmethod
    def update_experiment_settings(args):
        experiment_settings = get_spec(args.experiment_settings)
        if args.tuner:
            experiment_settings["SearchSettings"]["TunerName"] = args.tuner
        if args.trials:
            found_budget = False
            budgets = experiment_settings.get("Budget", [])
            for budget in budgets:
                if budget["Type"] == "ConfigurationCount":
                    budget["BudgetValue"] = args.trials
                    found_budget = True
            if len(budgets) == 0 or not found_budget:
                budgets.append({
                    "BudgetValue": args.trials,
                    "Type": "ConfigurationCount"
                })
                experiment_settings["Budget"] = budgets
        if args.logging:
            experiment_settings["General"]["LoggingLevel"] = args.logging
        if args.output_format:
            experiment_settings["General"]["OutputFormat"] = args.output_format
        if args.benchmarks:
            if args.benchmarks[0].lower() == "all":
                benchmarks = ["GEMM","nbody", "convolution", "dedisp", "TRIAD", "hotspot", "MD5Hash", "expdist"]
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
            args.json = os.path.join(".", "benchmarks", benchmark, f"{benchmark}-CAFF.json")
            args.benchmark = benchmark
            for tuner in tuners:
                print(f"Running {tuner} with {args}")
                self.runner_dict[tuner.lower()](args)
