from __future__ import print_function

import argparse
import logging
from builtins import str

import opentuner
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import (ConfigurationManipulator, EnumParameter)

from src.json import reader, kernel_specification

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument('--size', type=int, default=1, help='data set size for benchmark')
parser.add_argument('--precision', type=int, default=32, help='precision for benchmark')
parser.add_argument('--json', type=str, default="MD-CAFF.json", help='location of T1 json file')


class OpenTunerT(MeasurementInterface):
    def __init__(self, *pargs, **kwargs):
        super(OpenTunerT, self).__init__(*pargs, **kwargs)
        self.benchmark_config = {
            "PRECISION": self.args.precision
        }
        self.spec = kernel_specification.get_spec(self.args.json)
        self.config_space = self.spec["configurationSpace"]
        self.kernel_spec = self.spec["kernelSpecification"]

    def run(self, desired_result, input, limit):
        tuning_config = desired_result.configuration.data
        # print("Tuning config:", tuning_config)
        val = reader.core(self.args.json, self.benchmark_config, tuning_config)
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


if __name__ == '__main__':
    args = parser.parse_args()
    OpenTunerT.main(args)
