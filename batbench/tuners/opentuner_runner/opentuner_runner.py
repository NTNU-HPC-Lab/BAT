from __future__ import print_function

import argparse
import time

import opentuner
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import (ConfigurationManipulator, EnumParameter)

from batbench.result.result import Result
from batbench.manager.manager import Manager


class OpenTunerT(MeasurementInterface):

    result_list = []

    def __init__(self, *pargs, **kwargs):
        pargs[0].no_dups = True
        super(OpenTunerT, self).__init__(*pargs, **kwargs)
        self.manager = Manager(self.args)
        self.result = Result()
        self.n_trials = self.manager.budget_trials
        self.current_trial = 0
        self.time_0 = time.time()

    def run(self, desired_result, input, limit):
        self.result.algorithm_time = time.time() - self.time_0
        if self.current_trial == self.n_trials:
            self.save_final_config(None)

        tuning_config = desired_result.configuration.data
        #self.result.framework_time = time.time() - self.time_0
        self.result.config = tuning_config
        prev_result = self.manager.run(tuning_config, self.result)
        if prev_result.validity != "KnownConstraintsViolated":
            self.current_trial += 1
        self.time_0 = time.time()
        self.result = Result(self.manager.problem.spec)
        return opentuner.resultsdb.models.Result(time=prev_result.objective) # type: ignore

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        for (name, values) in self.manager.config_space.get_parameters_pair():
            manipulator.add_parameter(EnumParameter(name, values))
        return manipulator

    def program_name(self):
        return self.manager.problem.spec["KernelSpecification"]["KernelName"]

    def program_version(self):
        return 1.0

    def save_final_config(self, config):
        """
        called at the end of autotuning with the best resultsdb.models.Configuration
        """
        self.manager.dataset.final_write_data()
        self.manager.finished()
        #print(self.manager.dataset.get_best())
        raise NotImplementedError()

def main():
    openparser = argparse.ArgumentParser(parents=opentuner.argparsers())
    openparser.add_argument('--json', type=str, default="./benchmarks/MD5Hash-CAFF.json",
                            help='location of T1 json file')
    openparser.add_argument('--testing', type=str, default=False,
                            help='If the execution is a test or not')

    args = openparser.parse_args()
    OpenTunerT.main(args)


if __name__ == "__main__":
    main()
