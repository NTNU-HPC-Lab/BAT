import argparse
import logging

log = logging.getLogger(__name__)


def run_opentuner(args):
    from src.tuners.opentuner.opentuner_runner import OpenTunerT
    time = 1    # Run the tuner for 5 seconds
    args2 = args
    args2.stop_after = time
    OpenTunerT.main(args2)


def run_optuna(args):
    from src.tuners.optuna.optuna_runner import Optuna
    optuna_runner = Optuna()
    print(optuna_runner.main(args))


def run_kerneltuner(args):
    from src.tuners.kerneltuner.kerneltuner_runner import KernelTuner
    kerneltuner_runner = KernelTuner()
    print(kerneltuner_runner.main(args))


runner_dict = {
    "opentuner": run_opentuner,
    "optuna": run_optuna,
    "kerneltuner": run_kerneltuner,
}


def main():
    try:
        import opentuner
        parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    except ModuleNotFoundError:
        parser = argparse.ArgumentParser()
    parser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')
    parser.add_argument('--tuner', type=str, default=['kerneltuner'], nargs='+', help='Which tuner to use')
    parser.add_argument('--json', type=str, default="src/benchmarks/MD5Hash/MD5Hash-CAFF.json", help='Path to T1-compliant JSON')
    parser.add_argument('--benchmark', type=str, default=False, nargs='+', help='Which benchmarks to run')
    parser.add_argument('--gpu_name', type=str, default='A4000', help='The CUDA GPU to run on')
    args = parser.parse_args()

    for tuner in args.tuner:
        if tuner is not None:
            print("Running {} with {}".format(tuner, args))
            runner_dict[tuner](args)


if __name__ == '__main__':
    main()
