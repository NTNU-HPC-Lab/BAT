import optuna

from src.reader import T1_specification, reader


class Optuna:
    spec = None
    testing = False

    def objective(self, trial):
        tuning_config = self.parse_tuning_parameters(self.spec["configurationSpace"]["tuningParameters"], trial)
        return reader.core(self.args.json, self.spec["kernelSpecification"]["benchmarkConfig"], tuning_config, self.args.testing)

    def parse_tuning_parameters(self, tuning_parameters, trial):
        tuning_config = {}
        for param in tuning_parameters:
            tuning_config[param["name"]] = trial.suggest_categorical(param["name"], eval(str(param["values"])))
        return tuning_config

    def main(self, args):
        self.args = args
        self.spec = T1_specification.get_spec(args.json)
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=100)

        return study.best_params

