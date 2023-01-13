import datetime


class Result:
    def __init__(self, config=[0], objective=10000.0, compile_time=0, runtimes=[0], algorithm_time=0, framework_time=0, total_time=0, arg_time=0, times={}):
        self.times = times
        self.config = config
        self.compile_time = compile_time
        self.runtimes = runtimes
        self.algorithm_time = algorithm_time
        self.framework_time = framework_time
        self.arg_time = arg_time
        self.total_time = total_time
        self.correctness = 1
        self.validity = "Correct"
        self.error = "no error"
        self.objective = objective
        #self.objective = 10000 #TODO: Find a better value. float("inf") was not allowed by JSON parsers. But that might still be the best option, with a wrapper.
        self.timestamp = datetime.datetime.now()

    def isValid(self):
        return self.validity == "Correct" and self.objective > 0

    def __str__(self):
        return f"Timestamp: {self.timestamp},\nConfig: {self.config}\nValidity: {self.validity}\nObjective: {self.objective:.2E}"

    def calculate_time(self):
        self.total_time = (datetime.datetime.now() - self.timestamp).total_seconds()
        self.framework_time = self.total_time - self.compile_time - sum(self.runtimes) - self.algorithm_time

    def serialize(self):
        d = {}
        d["timestamp"] = str(self.timestamp)
        d["config"] = self.config
        d["correctness"] = self.correctness
        d["validity"] = self.validity
        d["objective"] = self.objective
        if self.error:
            d["error"] = str(self.error)
        if (len(self.times)):
            d["times"] = self.times
        else:
            d["times"] = {}
            d["times"]["total_time"] = self.total_time
            d["times"]["compile_time"] = self.compile_time
            d["times"]["arg_time"] = self.arg_time
            #d["times"]["runtimes"] = mean(self.runtimes) if len(self.runtimes) else 0
            d["times"]["runtimes"] = self.runtimes
            d["times"]["algorithm_time"] = self.algorithm_time
            d["times"]["framework_time"] = self.framework_time
        return d

