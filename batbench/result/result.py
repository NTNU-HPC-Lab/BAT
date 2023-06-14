import datetime
from typing import Dict, List, Optional


class Result:
    """
    A class representing the result of a benchmark run.

    Attributes:
        times (Dict[str, float]): A dictionary containing the times for different parts of the benchmark run.
        config (Dict): A dictionary containing the configuration used for the benchmark run.
        runtimes (List[float]): A list containing the runtimes for each iteration of the benchmark.
        compile_time (float): The time it took to compile the benchmark.
        algorithm_time (float): The time it took to run the algorithm part of the benchmark.
        framework_time (float): The time it took to run the framework part of the benchmark.
        arg_time (float): The time it took to parse the command line arguments for the benchmark.
        total_time (float): The total time it took to run the benchmark.
        correctness (float): A measure of the correctness of the benchmark run.
        validity (str): A string indicating whether the benchmark run was correct or not.
        error (str): A string containing any error messages that occurred during the benchmark run.
        objective (float): The objective value for the benchmark run.
        timestamp (datetime.datetime): The timestamp for when the benchmark run was completed.
    """
    def __init__(self, config: Optional[Dict] = None, objective: float = 10000.0,
                 compile_time: float = 0, runtimes: Optional[List[float]] = None,
                 algorithm_time: float = 0, framework_time: float = 0, total_time: float = 0,
                 arg_time: float = 0, times: Optional[Dict] = None):
        self.times = times if times is not None else {}
        self.config = config if config is not None else {}
        self.runtimes = runtimes if runtimes is not None else [0]
        self.compile_time = compile_time
        self.algorithm_time = algorithm_time
        self.framework_time = framework_time
        self.arg_time = arg_time
        self.total_time = total_time
        self.correctness = 1.0
        self.validity = "Correct"
        self.error = "no error"
        self.objective = objective
        self.timestamp = datetime.datetime.now()
        #self.objective = 10000 #TODO: Find a better value. float("inf") was not allowed by JSON parsers.
                                # But that might still be the best option, with a wrapper.

    def is_valid(self):
        return self.validity == "Correct" and self.objective > 0

    def calculate_time(self, timestamp):
        self.total_time = (datetime.datetime.now() - timestamp).total_seconds()

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

    def __str__(self):
        return f"""Timestamp: {self.timestamp},\nConfig: {self.config}\n
        Validity: {self.validity}\nObjective: {self.objective:.2E}"""

    def __repr__(self) -> str:
        return f"""Result({self.config}, {self.objective}, {self.compile_time},
                {self.runtimes}, {self.algorithm_time}, {self.arg_time}, {self.times})"""
