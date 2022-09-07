import argparse
import logging

log = logging.getLogger(__name__)


def run_opentuner(args):
    from src.tuners.opentuner.opentuner_runner import OpenTunerT
    OpenTunerT.main(args)


def run_optuna(args):
    from src.tuners.optuna.optuna_runner import Optuna
    #import cProfile, pstats
    #profiler = cProfile.Profile()
    #profiler.enable()
    optuna_runner = Optuna()
    print(optuna_runner.main(args))
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats()


def run_kerneltuner(args):
    from src.tuners.kerneltuner.kerneltuner_runner import KernelTuner
    kerneltuner_runner = KernelTuner()
    print(kerneltuner_runner.main(args))


runner_dict = {
    "opentuner": run_opentuner,
    "optuna": run_optuna,
    "kerneltuner": run_kerneltuner,
}


def add_standard_arguments_to_parser(parser):
    parser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')
    parser.add_argument('--tuner', type=str, default=['kerneltuner'], nargs='+', help='Which tuners to use')
    parser.add_argument('--benchmarks', type=str, default=["MD5Hash"], nargs='+', help='Name of T1-compliant JSON')
    parser.add_argument('--json', type=str, default="src/benchmarks/MD5Hash/MD5Hash-CAFF.json", help='Path to T1-compliant JSON')
    parser.add_argument('--trials', type=int, default=10, help='Path to T1-compliant JSON')
    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_standard_arguments_to_parser(parser)

    args_, _ = parser.parse_known_args()
    if "opentuner" in args_.tuner:
        import opentuner
        parser = argparse.ArgumentParser(parents=opentuner.argparsers())
        parser = add_standard_arguments_to_parser(parser)
    if "kerneltuner" in args_.tuner:
        parser.add_argument('--gpu_name', type=str, default='A4000', help='The CUDA GPU to run on')

    args = parser.parse_args()
    if args.benchmarks[0].lower() == "all":
        args.benchmarks = ["MD5Hash", "MD", "TRIAD", "builtin_vectors", "nbody", "Reduction"]
    for benchmark in args.benchmarks:
        args.json = "./src/benchmarks/{}/{}-CAFF.json".format(benchmark, benchmark)
        for tuner in args.tuner:
            if tuner is not None:
                print("Running {} with {}".format(tuner, args))
                runner_dict[tuner](args)


if __name__ == '__main__':
    main()
