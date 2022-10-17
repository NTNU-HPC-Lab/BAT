import time

from src.manager import Manager
from src.result import Result

from ConfigSpace import Categorical, ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

class SMAC3:
    def objective(self, config):
        self.result = Result(self.manager.spec)
        self.result = self.manager.run(config._values, self.result)
        return min(self.result.objective, 1000)

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

        smac = SMAC4BB(scenario=scenario, tae_runner=self.objective)
        best_found_config = smac.optimize()
        self.manager.write()
        return best_found_config


