import itertools
from typing import List, Dict, Any, Iterator, Tuple
class ConfigSpace:
    parameters = {}
    constraints = []

    def __init__(self, spec_config: Dict[str, Any] = {}) -> None:
        if "TuningParameters" in spec_config:
            for param in spec_config["TuningParameters"]:
                self.add_enum(param["Name"], eval(str(param["Values"])))
        if "Conditions" in spec_config:
            for expr in spec_config["Conditions"]:
                self.add_constraint(expr["Expression"], expr["Parameters"])

    def add_enum(self, key: str, enum: List[Any]) -> 'ConfigSpace':
        self.parameters[key] = enum
        return self

    def add_constraint(self, expr: str, params: List[str]) -> 'ConfigSpace':
        self.constraints.append({"Expression": expr, "Parameters": params})
        return self

    def get_product(self) -> Iterator[Tuple[Any, ...]]:
        return itertools.product(*(self.parameters.values()))

    def get_constraints(self) -> List[Dict[str, Any]]:
        return self.constraints

    def get_parameters(self) -> Dict[str, List[Any]]:
        return self.parameters

    def get_parameters_pair(self) -> Iterator[Tuple[str, List[Any]]]:
        return self.parameters.items()

    def make_constrained_iter(self) -> Iterator[Tuple[Any, ...]]:
        if not self.constraints: 
            return self.get_product()
        return filter(self.check_constraints, self.get_product())

    def check_constraints(self, config: Tuple[Any, ...]) -> bool:
        tuning_config = dict(zip(self.parameters.keys(), config))
        for expr in self.constraints:
            if not eval(expr["Expression"], tuning_config):
                return False
        return True

    def __contains__(self, config: Tuple[Any, ...]) -> bool:
        params = list(self.parameters.values())
        for i in range(len(config)):
            if config[i] not in params[i]:
                return False

        return self.check_constraints(config)

    def __iter__(self) -> 'ConfigSpace':
        return iter(self.make_constrained_iter())

    def __next__(self) -> Tuple[Any, ...]:
        return next(self.iter_list)

    def __str__(self) -> str:
        return "{}, {}".format(self.get_parameters_pair(), self.get_constraints())

