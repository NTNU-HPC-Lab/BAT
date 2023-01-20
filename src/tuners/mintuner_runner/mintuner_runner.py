import time

import copy

from src.manager import Manager
from src.result import Result

class MinTuner:

    def run(self, conf, result):
        tuning_config = {}
        for i, key in enumerate(self.manager.config_space.get_parameters().keys()):
            tuning_config[key] = conf[i]
        #result.config = copy.deepcopy(tuning_config)
        return self.manager.run(tuning_config, result)


    def exhaustive_search(self, args):
        t0 = time.time()
        i = 0
        result = Result()
        for l in self.manager.config_space:
            if i >= self.manager.budget_trials:
                break
            result.algorithm_time = time.time() - t0
            self.run(l, result)
            if self.result.validity != "KnownConstraintsViolated":
                i += 1
            t0 = time.time()
            result = Result()

    def random_search(self, args):
        from random import randint
        t0 = time.time()
        i = 0
        result = Result()
        while i < self.manager.budget_trials:
            config = {}
            for key, values in self.manager.config_space.get_parameters_pair():
                config[key] = values[randint(0, len(values)-1)]

            result.algorithm_time = time.time() - t0
            self.run(list(config.values()), result)
            if self.result.validity != "KnownConstraintsViolated":
                i += 1
            t0 = time.time()
            result = Result()

    def main(self, args):
        self.manager = Manager(args)

        #self.exhaustive_search(args)
        self.random_search(args)

        self.manager.dataset.final_write_data()
        best = self.manager.dataset.get_best()
        self.manager.finished()
        return best
