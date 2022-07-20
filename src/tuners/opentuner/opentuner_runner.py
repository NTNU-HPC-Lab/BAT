from __future__ import print_function

import argparse
from builtins import str

import opentuner
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import (ConfigurationManipulator, EnumParameter)

from src.readers.python.cuda.cupy_reader import CupyReader


class OpenTunerT(MeasurementInterface):
    def __init__(self, *pargs, **kwargs):
        super(OpenTunerT, self).__init__(*pargs, **kwargs)
        self.reader = CupyReader(self.args.json)
        self.spec = self.reader.get_spec(self.args.json)
        self.config_space = self.spec["configurationSpace"]
        self.kernel_spec = self.spec["kernelSpecification"]

    def run(self, desired_result, input, limit):
        tuning_config = desired_result.configuration.data
        val = self.reader.run(tuning_config, self.args.testing)
        return opentuner.resultsdb.models.Result(time=val)

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        for param in self.config_space["tuningParameters"]:
            manipulator.add_parameter(EnumParameter(param["name"], eval(str(param["values"]))))
        return manipulator

    def program_name(self):
        return self.kernel_spec["kernelName"]

    def program_version(self):
        return self.spec["general"]["formatVersion"]

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
