import time
from random import randint

from batbench.manager.manager import Manager
from batbench.result.result import Result

class MinTuner:

    def run(self, conf, result, config_space, run):
        tuning_config = {}
        for i, key in enumerate(config_space.get_parameters().keys()):
            tuning_config[key] = conf[i]
        result.config = tuning_config
        return run(tuning_config, result)


    def exhaustive_search(self, budget_trials, config_space, run):
        time_0 = time.time()
        i = 0
        result = Result()
        for config in config_space:
            if i >= budget_trials:
                break
            result.algorithm_time = time.time() - time_0
            result = self.run(config, result, config_space, run)
            if result.validity != "KnownConstraintsViolated":
                i += 1
            time_0 = time.time()
            result = Result()

    def random_search(self, budget_trials, config_space, run):
        time_0 = time.time()
        i = 0
        result = Result()
        while i < budget_trials:
            config = {}
            for key, values in config_space.get_parameters_pair():
                config[key] = values[randint(0, len(values)-1)]

            result.algorithm_time = time.time() - time_0
            result = self.run(list(config.values()), result, config_space, run)
            if result.validity != "KnownConstraintsViolated":
                i += 1
            time_0 = time.time()
            result = Result()

    def main(self, args):
        manager = Manager(args)

        #self.exhaustive_search(self.manager.budget_trials, self.manager.config_space)
        self.random_search(manager.budget_trials, manager.config_space, manager.run)

        manager.dataset.final_write_data()
        best = manager.dataset.get_best()
        manager.finished()
        return best
