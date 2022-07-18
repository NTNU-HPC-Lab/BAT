import argparse
import logging

import opentuner

from src.tuners.opentuner.opentuner_runner import OpenTunerT
from src.tuners.optuna.optuna_runner import Optuna

log = logging.getLogger(__name__)


def run_opentuner(args):
    time = 1  # Run the tuner for 5 seconds
    args2 = args
    args2.stop_after = time
    OpenTunerT.main(args2)


def run_optuna(args):
    optuna_runner = Optuna()
    print(optuna_runner.main(args))


runner_dict = {
    "opentuner": run_opentuner,
    "optuna": run_optuna
}


def main():
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    parser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')
    parser.add_argument('--tuner', type=str, default=False, nargs='+', help='Which tuner to use')
    parser.add_argument('--json', type=str, default=False, help='Path to T1-compliant JSON')
    parser.add_argument('--benchmark', type=str, default=False, nargs='+', help='Which benchmarks to run')
    args = parser.parse_args()

    for tuner in args.tuner:
        if tuner is not None:
            print("Running {} with {}".format(tuner, args))
            runner_dict[tuner](args)


if __name__ == '__main__':
    main()
