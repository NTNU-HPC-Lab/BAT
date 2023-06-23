from batbench.config_space.config_space import ConfigSpace

def test_check_constraints():
    config_space = ConfigSpace()
    config_space.add_enum("x", [2, 3, 4])
    config_space.add_enum("y", [1, 2, 3])
    config_space.add_constraint("x > y", ["x", "y"])

    # Change to a configuration that satisfies the constraints
    assert config_space.check_constraints((3, 2))

def test_make_constrained_iter():
    config_space = ConfigSpace()
    config_space.add_enum("x", [2, 3, 4])
    config_space.add_enum("y", [1, 2, 3])
    config_space.add_constraint("x > y", ["x", "y"])

    constrained_iter = config_space.make_constrained_iter()
    # The first configuration that satisfies the constraints is (3, 2)
    assert next(constrained_iter) == (2, 1)
    assert next(constrained_iter) == (3, 1)
    assert next(constrained_iter) == (3, 2)
    assert next(constrained_iter) == (4, 1)

def test_config_space_contains():
    config_space = ConfigSpace()
    config_space.add_enum("x", [2, 3, 4])
    config_space.add_enum("y", [1, 2, 3])
    config_space.add_constraint("x > y", ["x", "y"])
    # This configuration satisfies the constraints
    assert (3, 2) in config_space

def test_next():
    config_space = ConfigSpace()
    config_space.add_enum("x", [2, 3, 4])
    config_space.add_enum("y", [1, 2, 3])
    config_space.add_constraint("x > y", ["x", "y"])
    # The first configuration that satisfies the constraints is (3, 2)
    assert next(config_space) == (2, 1)
    assert next(config_space) == (3, 1)
    assert next(config_space) == (3, 2)
    assert next(config_space) == (4, 1)
