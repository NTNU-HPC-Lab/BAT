import json


def print_json(j):
    print(json.dumps(j, indent=4, sort_keys=True))