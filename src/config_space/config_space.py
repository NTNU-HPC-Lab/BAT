import itertools

class ConfigSpace:
    parameters = {}
    constraints = []

    def __init__(self, spec_config):
        for param in spec_config["TuningParameters"]:
            self.add_enum(param["Name"], eval(str(param["Values"])))
        if spec_config.get("Conditions"):
            for expr in spec_config["Conditions"]:
                self.add_constraint(expr)
        self.sort_parameters()

    def sort_parameters(self):
        self.parameters = dict(sorted(self.parameters.items()))

    def add_enum(self, key, enum):
        self.parameters[key] = enum
        self.sort_parameters()

    def add_constraint(self, expr):
        self.constraints.append(expr)

    def get_product(self):
        return itertools.product(*(self.parameters.values()))

    def get_constraints(self):
        return self.constraints

    def get_parameters(self):
        return self.parameters

    def get_parameters_pair(self):
        return self.parameters.items()

    def make_constrained_iter(self):
        if len(self.constraints) == 0: return self.get_product()
        return itertools.filterfalse(lambda x: not self.check_constraints(x), self.get_product())

    def check_constraints(self, config):
        tuning_config = dict(zip(self.parameters.keys(), config))
        for expr in self.constraints:
            if not eval(expr["Expression"], tuning_config):
                return False
        return True

    def __contains__(self, config):
        params = list(self.parameters.values())
        for i in range(len(config)):
            if config[i] not in params[i]:
                return False

        return self.check_constraints(config)

    def __iter__(self):
        self.iter_list = self.make_constrained_iter()
        return self

    def __next__(self):
        return next(self.iter_list)

    def __str__(self):
        return "{}, {}".format(self.get_parameters_pair(), self.get_constraints())

