import datetime
from typing import Dict, List


class Result:
    def __init__(self, config: List[int] = [0], objective: float = 10000.0, compile_time: float = 0, 
                 runtimes: List[float] = [0], algorithm_time: float = 0, framework_time: float = 0, total_time: float = 0, arg_time: float = 0, times: Dict = {}):
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
        self.timestamp = datetime.datetime.now()
        #self.objective = 10000 #TODO: Find a better value. float("inf") was not allowed by JSON parsers. But that might still be the best option, with a wrapper.

    def isValid(self):
        return self.validity == "Correct" and self.objective > 0

    def __str__(self):
        return f"Timestamp: {self.timestamp},\nConfig: {self.config}\nValidity: {self.validity}\nObjective: {self.objective:.2E}"
    
    def __repr__(self) -> str:
        return f"Result({self.config}, {self.objective}, {self.compile_time}, {self.runtimes}, {self.algorithm_time}, {self.arg_time}, {self.times})"


    def calculate_time(self, timestamp):
        self.total_time = (datetime.datetime.now() - timestamp).total_seconds()
        #self.framework_time = self.total_time - self.compile_time - sum(self.runtimes) - self.algorithm_time

    def serialize(self) -> Dict:
        return {
            "timestamp": str(self.timestamp),
            "config": self.config,
            "correctness": self.correctness,
            "validity": self.validity,
            "objective": self.objective,
            "error": str(self.error) if self.error else None,
            "times": self.times if len(self.times) else {
                "total_time": self.total_time,
                "compile_time": self.compile_time,
                "arg_time": self.arg_time,
                "runtimes": self.runtimes,
                "algorithm_time": self.algorithm_time,
                "framework_time": self.framework_time
            }
        }

