import argparse
import optuna

from src.readers.python.T1_specification import core
from src.readers.python.cuda.cupy_reader import CupyReader


class Optuna:
    spec = None
    testing = False
    reader = None

    def objective(self, trial):
        tuning_config = self.parse_tuning_parameters(self.spec["configurationSpace"]["tuningParameters"], trial)
        return core(self.args.json, tuning_config, self.args.testing)

    def parse_tuning_parameters(self, tuning_parameters, trial):
        tuning_config = {}
        for param in tuning_parameters:
            tuning_config[param["name"]] = trial.suggest_categorical(param["name"], eval(str(param["values"])))
        return tuning_config

    def main(self, args):
        self.args = args
        cupy_reader = CupyReader(args.json)
        # self.reader = CupyReader(self.args.json)
        self.reader = cupy_reader
        self.spec = cupy_reader.get_spec(args.json)
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=100)

        return study.best_params


def main():
    optunaparser = argparse.ArgumentParser()
    optunaparser.add_argument('--json', type=str, default="./benchmarks/MD5Hash-CAFF.json",
                              help='location of T1 json file')
    optunaparser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')

    args = optunaparser.parse_args()

    optuna_runner = Optuna()
    print(optuna_runner.main(args))


if __name__ == "__main__":
    main()
