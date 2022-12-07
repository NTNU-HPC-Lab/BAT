import time

from src.manager import Manager
from src.result import Result

from ConfigSpace import Categorical, ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

class SMAC3:
    def objective(self, config):
        self.result.algorithm_time = time.time() - self.t0
        prev_result = self.manager.run(config._values, self.result)
        self.t0 = time.time()
        self.result = Result(self.manager.spec)
        return prev_result.objective

    def main(self, args):
        self.manager = Manager(args)
        n_trials = self.manager.search_spec["Budget"]["BudgetValue"]
        cs = ConfigurationSpace(seed=1234)
        hyperparams = []
        for (name, values) in self.manager.config_space.get_parameters_pair():
            hyperparams.append(Categorical(name, items=values))

        cs.add_hyperparameters(hyperparams)

        scenario = Scenario({
            "run_obj": "quality",  # Optimize quality (alternatively runtime)
            "runcount-limit": n_trials,  # Max number of function evaluations (the more the better)
            "cs": cs,
        })

        self.t0 = time.time()
        self.result = Result(self.manager.spec)
        smac = SMAC4BB(scenario=scenario, tae_runner=self.objective)
        smac.optimize()
        self.manager.write()
        return self.manager.dataset.get_best()


