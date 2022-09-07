from __future__ import print_function

import argparse
from builtins import str

import opentuner
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import (ConfigurationManipulator, EnumParameter)

from src.result import Result
from src.manager import Manager


class OpenTunerT(MeasurementInterface):

    result_list = []

    def __init__(self, *pargs, **kwargs):
        super(OpenTunerT, self).__init__(*pargs, **kwargs)
        self.manager = Manager(self.args)
        self.result = Result(self.manager.spec)

    def run(self, desired_result, input, limit):
        tuning_config = desired_result.configuration.data
        self.result = Result(self.manager.spec)
        self.result.config = tuning_config
        self.result = self.manager.run(tuning_config, self.result)

        return opentuner.resultsdb.models.Result(time=self.result.objective)

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        for (name, values) in self.manager.config_space.get_parameters_pair():
            manipulator.add_parameter(EnumParameter(name, values))
        return manipulator

    def program_name(self):
        return self.manager.spec["kernelSpecification"]["kernelName"]

    def program_version(self):
        return self.manager.spec["general"]["formatVersion"]

    def save_final_config(self, configuration):
        """
        called at the end of autotuning with the best resultsdb.models.Configuration
        """
        print("Final configuration", configuration.data)


def main():
    openparser = argparse.ArgumentParser(parents=opentuner.argparsers())
    openparser.add_argument('--json', type=str, default="./benchmarks/MD5Hash-CAFF.json",
                            help='location of T1 json file')
    openparser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')

    args = openparser.parse_args()
    OpenTunerT.main(args)


if __name__ == "__main__":
    main()