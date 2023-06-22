import itertools
from typing import List, Dict, Any, Iterator, Tuple
class ConfigSpace:
    """
    A class representing a configuration space for tuning parameters.

    A configuration space is a set of parameters and constraints that 
    define a search space for tuning parameters.
    """
    parameters = {}
    constraints = []
    default_config = {}

    def __init__(self, spec_config: Dict[str, Any] = None) -> None:

        if spec_config is None:
            spec_config = {}
        if "TuningParameters" in spec_config:
            for param in spec_config["TuningParameters"]:
                self.add_enum(param["Name"], eval(str(param["Values"])))
                self.default_config[param["Name"]] = param["Default"]
        if "Conditions" in spec_config:
            for expr in spec_config["Conditions"]:
                self.add_constraint(expr["Expression"], expr["Parameters"])

        self.iter_list = self.make_constrained_iter()

    def add_enum(self, key: str, enum: List[Any]) -> 'ConfigSpace':
        """
        Adds an enumerated parameter to the configuration space.

        :param key: A string representing the name of the parameter.
        :type key: str
        :param enum: A list of possible values for the parameter.
        :type enum: List[Any]
        :return: The ConfigSpace object with the added parameter.
        :rtype: ConfigSpace
        """
        self.parameters[key] = enum
        return self

    def add_constraint(self, expr: str, params: List[str]) -> 'ConfigSpace':
        """
        Adds a constraint to the configuration space.

        :param expr: A string representing the constraint expression.
        :type expr: str
        :param params: A list of parameter names used in the constraint expression.
        :type params: List[str]
        :return: The ConfigSpace object with the added constraint.
        :rtype: ConfigSpace
        """
        self.constraints.append({"Expression": expr, "Parameters": params})
        return self

    def get_product(self) -> Iterator[Tuple[Any, ...]]:
        """
        Returns an iterator over the Cartesian product of the parameter values.

        :return: An iterator over the Cartesian product of the parameter values.
        :rtype: Iterator[Tuple[Any, ...]]
        """
        return itertools.product(*(self.parameters.values()))

    def get_constraints(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries representing the constraints in the configuration space.

        :return: A list of dictionaries representing the constraints in the configuration space.
        :rtype: List[Dict[str, Any]]
        """
        return self.constraints

    def get_parameters(self) -> Dict[str, List[Any]]:
        """
        Returns a dictionary representing the parameters in the configuration space.

        :return: A dictionary representing the parameters in the configuration space.
        :rtype: Dict[str, List[Any]]
        """
        return self.parameters

    def get_parameters_pair(self) -> Iterator[Tuple[str, List[Any]]]:
        """
        Returns an iterator over the parameters in the configuration space, 
        where each parameter is represented as a tuple containing the parameter name and 
        a list of its possible values.

        :return: An iterator over the parameters in the configuration space.
        :rtype: Iterator[Tuple[str, List[Any]]]
        """
        return iter(self.parameters.items())

    def make_constrained_iter(self) -> Iterator[Tuple[Any, ...]]:
        """
        Returns an iterator over the Cartesian product of the parameter values that 
        satisfy the constraints.

        :return: An iterator over the Cartesian product of the parameter values that satisfy the constraints.
        :rtype: Iterator[Tuple[Any, ...]]
        """
        if not self.constraints:
            return self.get_product()
        return filter(self.check_constraints, self.get_product())

    def get_default_config(self) -> Dict:
        return self.default_config

    def check_constraints(self, config: Tuple[Any, ...]) -> bool:
        """
        Checks if a given configuration satisfies all constraints in the configuration space.

        :param config: A tuple representing a configuration to be checked.
        :type config: Tuple[Any, ...]
        :return: True if the configuration satisfies all constraints, False otherwise.
        :rtype: bool
        """
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

    def __next__(self) -> Tuple[Any, ...]:
        return next(self.iter_list)

    def __str__(self) -> str:
        return f"{self.get_parameters_pair()}, {self.get_constraints()}"
