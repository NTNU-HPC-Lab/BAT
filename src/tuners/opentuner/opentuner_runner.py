from __future__ import print_function

from builtins import str

import opentuner
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import (ConfigurationManipulator, EnumParameter)

from src.reader import reader, T1_specification


class OpenTunerT(MeasurementInterface):
    def __init__(self, *pargs, **kwargs):
        super(OpenTunerT, self).__init__(*pargs, **kwargs)
        self.spec = T1_specification.get_spec(self.args.json)
        self.benchmark_config = self.spec["kernelSpecification"]["benchmarkConfig"]
        self.config_space = self.spec["configurationSpace"]
        self.kernel_spec = self.spec["kernelSpecification"]

    def run(self, desired_result, input, limit):
        tuning_config = desired_result.configuration.data
        val = reader.core(self.args.json, self.benchmark_config, tuning_config, self.args.testing)
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
