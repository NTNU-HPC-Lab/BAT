import argparse
import logging

import opentuner

from src.tuners.opentuner.opentuner_runner import OpenTunerT
from src.tuners.optuna.optuna_runner import Optuna

log = logging.getLogger(__name__)


def main(tuner):
    if tuner == "opentuner":
        openparser = argparse.ArgumentParser(parents=opentuner.argparsers())
        openparser.add_argument('--size', type=int, default=1, help='data set size for benchmark')
        openparser.add_argument('--precision', type=int, default=32, help='precision for benchmark')
        openparser.add_argument('--json', type=str, default="./benchmarks/MD5Hash-CAFF.json", help='location of T1 json file')
        openparser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')

        args = openparser.parse_args()

        print("Running", args.json)
        OpenTunerT.main(args)
    if tuner == "optuna":
        optunaparser = argparse.ArgumentParser()
        optunaparser.add_argument('--size', type=int, default=1, help='data set size for benchmark')
        optunaparser.add_argument('--json', type=str, default="./benchmarks/MD5Hash-CAFF.json",
                                  help='location of T1 json file')
        optunaparser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')

        args = optunaparser.parse_args()

        optuna_runner = Optuna()
        print(optuna_runner.main(args))


if __name__ == '__main__':
    # main("opentuner")
    main("optuna")
