import argparse
import logging

from src.manager import ExperimentManager

log = logging.getLogger(__name__)


def add_standard_arguments_to_parser(parser):
    #parser.add_argument('--testing', type=str, default=False, help='If the execution is a test or not')
    parser.add_argument('--tuner', type=str, default='kerneltuner', help='Which tuners to use')
    parser.add_argument('--benchmarks', type=str, default=None, nargs='+', help='Name of T1-compliant JSON')
    #parser.add_argument('--json', type=str, default=None, help='Path to T1-compliant JSON')
    #parser.add_argument('--output-format', type=str, default=None, help='Which file format to use to store results')
    #parser.add_argument('--trials', type=int, default=None, help='Path to T1-compliant JSON')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbosity level for tuner output')
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
    ExperimentManager().start(args)


if __name__ == '__main__':
    main()
