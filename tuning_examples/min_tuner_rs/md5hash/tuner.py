#!/usr/bin/env python
from tuning_examples.common.helpers2 import save_results, create_parser, write_oadc
from tuning_examples.min_tuner.main import MinTuner

if __name__ == '__main__':
    args = create_parser()
    write_oadc(args.parse_args())
    save_results(MinTuner(args.parse_args()).run())

