import datetime

from statistics import mean


class Result:
    def __init__(self, spec, config=[0], compile_time=0, runtimes=[0], algorithm_time=0, framework_time=0, total_time=0):
        self.spec = spec
        self.benchmark = spec["General"]["BenchmarkName"]
        self.config = config
        self.compile_time = compile_time
        self.runtimes = runtimes
        self.algorithm_time = algorithm_time
        self.framework_time = framework_time
        self.total_time = total_time
        self.correctness = 1
        self.validity = "Correct"
        self.objective = float("inf")
        self.timestamp = datetime.datetime.now()

    def __repr__(self):
        return "Timestamp: {},\nBenchmark: {}\nConfig: {}\nValidity: {}\nObjective: {:.2E}\nCompile time: {:.2E}\nRuntime: {:.2E}\nSearch Algorithm: {:.2E}\nFramework time: {:.2E}".format(self.timestamp, self.benchmark, self.config, self.validity, self.objective, self.compile_time, mean(self.runtimes), self.algorithm_time, self.framework_time)

    def calculate_time(self):
        self.total_time = (datetime.datetime.now() - self.timestamp).total_seconds()
        print(self.total_time)
        self.framework_time = self.total_time - self.compile_time - sum(self.runtimes) - self.algorithm_time

    def serialize(self):
        d = {}
        d["timestamp"] = str(self.timestamp)
        d["benchmark"] = self.benchmark
        d["config"] = self.config
        d["correctness"] = self.correctness
        d["validity"] = self.validity
        d["objective"] = self.objective
        d["times"] = {}
        d["times"]["total_time"] = self.total_time
        d["times"]["compile_time"] = self.compile_time
        #d["times"]["runtimes"] = mean(self.runtimes) if len(self.runtimes) else 0
        d["times"]["runtimes"] = self.runtimes
        d["times"]["algorithm_time"] = self.algorithm_time
        d["times"]["framework_time"] = self.framework_time
        return d

