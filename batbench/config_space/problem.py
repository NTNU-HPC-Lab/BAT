from abc import ABC, abstractmethod


class Problem(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def get_args(self):
        ...

    @abstractmethod
    def run(self, tuning_config, result):
        ...
