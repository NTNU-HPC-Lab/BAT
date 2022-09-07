from src.readers.python.config_space import ConfigSpace
from src.readers.python.cuda.cupy_reader import get_spec


def main():
    spec = get_spec("./src/benchmarks/builtin_vectors/builtin_vectors-CAFF.json")
    cs = ConfigSpace(spec["configurationSpace"])
    cs.add_enum("Loop", [1, 2, 4])
    cs.add_enum("Block", [32, 64, 128])
    cs.add_enum("Test1", [i for i in range(1, 1000)])
    cs.add_enum("Test2", [i for i in range(1, 1000)])
    cs.add_constraint("Loop * Block <= 128")
    #for e in list(cs.get_product()):
    #    print(e)
    #for e in cs.get_constraints():
    #    print(e)

    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    # cs.generate_constrained_list()
    print((1024, 4, 32, 900, 100) in cs)
    print((1024, 4, 64, 900, 900) in cs)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

main()
