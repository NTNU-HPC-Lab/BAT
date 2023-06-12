from abc import ABC, abstractmethod, abstractproperty
from bat.config_space.config_space import ConfigSpace


class Problem(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def get_args(self):
        ...
    
    @abstractmethod
    def run(self, tuning_config, result):
        ...