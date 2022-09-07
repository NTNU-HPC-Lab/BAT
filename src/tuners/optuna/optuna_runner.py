import argparse
import optuna
import time

from src.readers.python.result import Result
from src.readers.python.config_space import ConfigSpace

from src.readers.python.cuda.cupy_reader import get_spec
from src.readers.python.cuda.cuda_kernel_runner import CudaKernelRunner

import json

class Optuna:

    total_results = []

    def objective(self, trial):
        self.result = Result(self.spec)
        tuning_config = self.get_next_tuning_config(trial)
        self.result.config = tuning_config
        self.result = self.runner.run(tuning_config, self.result, self.args.testing)
        # print(self.result)
        self.total_results.append(self.result)
        return self.result.objective

    def get_next_tuning_config(self, trial):
        tuning_config = {}
        t0 = time.time()
        for (name, values) in self.config_space.get_parameters_pair():
            tuning_config[name] = trial.suggest_categorical(name, values)

        self.result.algorithm_time = time.time() - t0
        return tuning_config

    def main(self, args):
        self.args = args
        self.spec = get_spec(args.json)
        self.config_space = ConfigSpace(self.spec["configurationSpace"])
        self.runner = CudaKernelRunner(self.spec, self.config_space)
        n_trials = args.trials
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=n_trials)
        self.result.write("optuna-results.json", self.total_results)
        return study.best_params


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    optunaparser = argparse.ArgumentParser()
    optunaparser.add_argument('--json', type=str, default="./benchmarks/MD5Hash-CAFF.json",
                              help='location of T1 json file')
    optunaparser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')

    args = optunaparser.parse_args()

    optuna_runner = Optuna()
    print(optuna_runner.main(args))


if __name__ == "__main__":
    main()
