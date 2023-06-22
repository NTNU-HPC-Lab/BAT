import argparse
import logging
from batbench.util import get_spec
from batbench.manager.experiment_manager import ExperimentManager

# move global imports to the top of your script
try:
    import opentuner
except ImportError:
    opentuner = None

log = logging.getLogger(__name__)

def add_standard_arguments_to_parser(parser):
    parser.add_argument('--tuner', type=str, default=["mintuner"], nargs='+',
                        help='Which tuners to use')
    parser.add_argument('--benchmarks', type=str, default=["GEMM"], nargs='+',
                        help='Name of T1-compliant JSON')
    parser.add_argument('--json', type=str, default=None, 
                        help='Path to T1-compliant JSON')
    parser.add_argument('--experiment-settings', type=str, default="experiment-settings.json",
                        help='Path to experiment settings JSON')
    parser.add_argument('--output-format', type=str, default=None,
                        help='Which file format to use to store results')
    parser.add_argument('--trials', type=int, default=10,
                        help='Path to T1-compliant JSON')
    parser.add_argument('--logging', type=str, default=None,
                        help='Verbosity level for tuner output')
    parser.add_argument('--cleanup', type=bool, default=False,
                        help='Whether or not to delete produced files after running.')
    return parser

def add_kerneltuner_arguments_to_parser(parser):
    parser.add_argument('--gpu_name', type=str, default='A4000', help='The CUDA GPU to run on')
    parser.add_argument('--cache', type=str, default='', help='The cache to use')
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser()
    add_standard_arguments_to_parser(parser)

    args_, _ = parser.parse_known_args()
    experiment_settings = get_spec(args_.experiment_settings)
    tuners = experiment_settings["SearchSettings"]["TunerName"]

    if "opentuner" in tuners or (args_.tuner is not None and "opentuner" in args_.tuner):
        if opentuner is not None:
            parser = argparse.ArgumentParser(parents=opentuner.argparsers())
            add_standard_arguments_to_parser(parser)
        else:
            log.error("opentuner module is not installed but is required.")

    if "kerneltuner" in tuners or (args_.tuner is not None and "kerneltuner" in args_.tuner):
        add_kerneltuner_arguments_to_parser(parser)

    return parser.parse_args()


def main():
    args = parse_arguments()
    ExperimentManager().start(args)


if __name__ == '__main__':
    main()
