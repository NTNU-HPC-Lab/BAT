import logging
import os

from ..util import get_spec, write_spec


# Create a custom logger
log = logging.getLogger(__name__)

class ExperimentManager:
    """
    The ExperimentManager class is responsible for managing experiments in BATBench.
    It provides functionality for updating experiment settings based on command line arguments,
    and running the specified tuner(s) on the specified benchmark(s).
    """
    def __init__(self):
        self.runner_dict = {
            "opentuner": self.run_opentuner,
            "optuna": self.run_optuna,
            "kerneltuner": self.run_kerneltuner,
            "smac": self.run_smac3,
            "smac3": self.run_smac3,
            "mintuner": self.run_mintuner,
            "ktt": self.run_ktt
        }

    @staticmethod
    def run_opentuner(args):
        from batbench.tuners.opentuner_runner import OpenTunerT
        try:
            print(OpenTunerT.main(args))
        except NotImplementedError:
            log.error("Terminated due to NotImplementedError in OpenTunerT.main")

    @staticmethod
    def run_kerneltuner(args):
        from batbench.tuners.kerneltuner_runner import KernelTuner
        print(KernelTuner().main(args))

    @staticmethod
    def run_mintuner(args):
        from batbench.tuners.mintuner_runner import MinTuner
        print(MinTuner().main(args))

    @staticmethod
    def run_optuna(args):
        from batbench.tuners.optuna_runner import Optuna
        print(Optuna().main(args))

    @staticmethod
    def run_smac3(args):
        from batbench.tuners.smac3_runner import SMAC3
        print(SMAC3().main(args))

    @staticmethod
    def run_ktt(args):
        raise NotImplementedError
        #print(tuners.KTTRunner().main(args))


    @staticmethod
    def update_experiment_settings(args):
        """
        Updates the experiment settings based on the command line arguments passed in.

        Args:
            args: The command line arguments.

        Returns:
            The updated experiment settings.
        """
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
                benchmarks = ["GEMM","nbody", "convolution", "dedisp", 
                              "TRIAD", "hotspot", "MD5Hash", "expdist"]
            else:
                benchmarks = args.benchmarks
            experiment_settings["BenchmarkConfig"]["Benchmarks"] = benchmarks
        write_spec(experiment_settings, args.experiment_settings)
        return experiment_settings

    def start(self, args):
        """
        Starts the experiment by updating the experiment settings based on the 
        command line arguments passed in,
        and then running the specified tuner(s) on the specified benchmark(s).

        Args:
            args: The command line arguments.

        Returns:
            None.
        """
        experiment_settings = self.update_experiment_settings(args)
        experiment_tuners = experiment_settings["SearchSettings"]["TunerName"]
        benchmarks = experiment_settings["BenchmarkConfig"]["Benchmarks"]

        if args.json:
            for tuner in experiment_tuners:
                print(f"Running {tuner} with {args}")
                self.runner_dict[tuner.lower()](args)
            return

        for benchmark in benchmarks:
            args.json = os.path.join(".", "benchmarks", benchmark, f"{benchmark}-CAFF.json")
            args.benchmark = benchmark
            for tuner in experiment_tuners:
                print(f"Running {tuner} with {args}")
                self.runner_dict[tuner.lower()](args)
