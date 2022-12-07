import time

from memory_profiler import profile
import copy

from src.manager import Manager
from src.result import Result

class MinTuner:

    def run(self, conf, result):
        tuning_config = {}
        for i, key in enumerate(self.manager.config_space.get_parameters().keys()):
            tuning_config[key] = conf[i]
        result.config = copy.deepcopy(tuning_config)
        return self.manager.run(tuning_config, result)

    def main(self, args):
        self.manager = Manager(args)

        t0 = time.time()
        for i, l in enumerate(list(self.manager.config_space.get_product())):
            if i >= self.manager.budget:
                break
            t1 = time.time()
            result = Result()
            result.algorithm_time = t1 - t0
            self.run(l, result)
            t0 = time.time()


        self.manager.write()
        return self.manager.dataset.get_best()
