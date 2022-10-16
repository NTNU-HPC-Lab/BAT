import argparse
import optuna
import time

from src.manager import Manager
from src.result import Result

class Optuna:

    def objective(self, trial):
        self.result = Result(self.manager.spec)
        tuning_config = self.get_next_tuning_config(trial)
        self.result = self.manager.run(tuning_config, self.result)
        return self.result.objective

    def get_next_tuning_config(self, trial):
        tuning_config = {}
        t0 = time.time()
        for (name, values) in self.manager.config_space.get_parameters_pair():
            tuning_config[name] = trial.suggest_categorical(name, values)

        self.result.algorithm_time = time.time() - t0
        return tuning_config

    def main(self, args):
        self.manager = Manager(args)
        n_trials = self.manager.search_spec["Budget"]["BudgetValue"]
        if self.manager.search_spec["General"]["LoggingLevel"] != "Debug":
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=n_trials)
        self.manager.write()
        return study.best_params


def main():

    optunaparser = argparse.ArgumentParser()
    optunaparser.add_argument('--json', type=str, default="./benchmarks/MD5Hash-CAFF.json",
                              help='location of T1 json file')
    optunaparser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')

    args = optunaparser.parse_args()


    optuna_runner = Optuna()

    if not args.verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(optuna_runner.main(args))


if __name__ == "__main__":
    main()
