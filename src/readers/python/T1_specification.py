import json

from src.readers.python.cuda.cupy_reader import CupyReader


def core(json_path, tuning_config, testing):
    with open(json_path, 'r') as f:
        r = json.load(f)

    reader = CupyReader(json_path)
    launch_config = reader.get_launch_config(tuning_config)
    return reader.run_kernel(launch_config, tuning_config)

