import argparse
import optuna
import time

from bat.manager.manager import Manager
from bat.result.result import Result

class Optuna:

    def objective(self, trial):
        tuning_config = self.get_next_tuning_config(trial)
        self.result.config = tuning_config

        self.result.algorithm_time = time.time() - self.t0
        prev_result = self.manager.run(tuning_config, self.result)
        self.result = Result()
        return prev_result.objective

    def get_next_tuning_config(self, trial):
        tuning_config = {}
        for (name, values) in self.manager.config_space.get_parameters_pair():
            tuning_config[name] = trial.suggest_categorical(name, values)

        return tuning_config

    def main(self, args):
        self.manager = Manager(args)
        n_trials = self.manager.budget_trials
        #if self.manager.problem.spec["General"]["LoggingLevel"] != "Debug":
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        search_space = {}
        for (name, values) in self.manager.config_space.get_parameters_pair():
            search_space[name] = values

        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
        self.t0 = time.time()
        self.result = Result()
        study.optimize(self.objective, n_trials=n_trials)
        self.manager.dataset.final_write_data()
        best = self.manager.dataset.get_best()
        self.manager.finished()
        return best


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
