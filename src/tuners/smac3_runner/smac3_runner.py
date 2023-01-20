import time

from src.manager import Manager
from src.result import Result

from ConfigSpace import Categorical, ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

class SMAC3:
    def objective(self, config):
        self.result.config = config._values
        self.result.algorithm_time = time.time() - self.t0
        prev_result = self.manager.run(config._values, self.result)
        self.t0 = time.time()
        self.result = Result()
        return prev_result.objective

    def main(self, args):
        self.manager = Manager(args)
        n_trials = self.manager.budget_trials
        cs = ConfigurationSpace(seed=1234)
        hyperparams = []
        for (name, values) in self.manager.config_space.get_parameters_pair():
            hyperparams.append(Categorical(name, items=values))

        #print(hyperparams)
        cs.add_hyperparameters(hyperparams)
        #print(cs)

        scenario = Scenario({
            "run_obj": "quality",  # Optimize quality (alternatively runtime)
            "runcount-limit": 10000,  # Max number of function evaluations (the more the better)
            #"n_restarts_optimizer": 2,
            "cs": cs,
        })

        self.t0 = time.time()
        self.result = Result()
        try:
            smac = SMAC4BB(scenario=scenario, tae_runner=self.objective)
            smac.optimize()
        finally:
            self.manager.dataset.final_write_data()
            best = self.manager.dataset.get_best()
            self.manager.finished()
            return best


