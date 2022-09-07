import json

from statistics import mean

class Result:
    def __init__(self, spec, config=[0], compile_time=0, runtimes=[0], algorithm_time=0, framework_time=0):
        self.spec = spec
        self.benchmark = spec["general"]["benchmarkName"]
        self.config = config
        self.compile_time = compile_time
        self.runtimes = runtimes
        self.algorithm_time = algorithm_time
        self.framework_time = framework_time
        self.correctness = 1
        self.validity = "Correct"
        self.objective = float("inf")

    def __repr__(self):
        return "Benchmark: {}\nConfig: {}\nValidity: {}\nObjective: {:.2E}\nCompile time: {:.2E}\nRuntime: {:.2E}\nSearch Algorithm: {:.2E}\nFramework time: {:.2E}".format(self.benchmark, self.config, self.validity, self.objective, self.compile_time, mean(self.runtimes), self.algorithm_time, self.framework_time)

    def serialize(self):
        d = {}
        d["benchmark"] = self.benchmark
        d["config"] = self.config
        d["correctness"] = self.correctness
        d["validity"] = self.validity
        d["objective"] = self.objective
        d["times"] = {}
        d["times"]["compile_time"] = self.compile_time
        d["times"]["runtimes"] = self.runtimes
        d["times"]["algorithm_time"] = self.algorithm_time
        d["times"]["framework_time"] = self.framework_time
        return d

    def write(self, filename, results=[]):
        if results == []: results = [self]
        dump_results = {"results": []}
        with open(filename, 'a') as f:
            for result in results:
                dump_results["results"].append(result.serialize())
            json.dump(dump_results, f, indent=4)
