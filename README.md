# T1 Reader


## Setup

### Prerequisites
Python3 and pip3
All the current tuners are python3-based tuners. C++ based tuners will be added soon.

The python path must be the source of the project.
One way to set this is to move to the root directory of this project and execute
```
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Readers
The Python-based T1-format reader requires the python library cupy to execute CUDA kernels from python.
This can be installed with
```
pip install -r src/readers/python/requirements.txt
```
The Python-based tuners that do not have native T1-support use this library to execute their configurations.


### Tuners
Each tuner can be installed with the requirements.txt file in their sub-folder src/tuners/*/requirements.txt. E.g.
```
pip install -r src/tuners/optuna/requirements.txt
```


## Run using only a specific tuner
### Optuna
```
python3 src/tuners/optuna/optuna_runner.py --json=./src/benchmarks/builtin_vectors/builtin_vectors.json
```
### Opentuner
```
python3 src/tuners/opentuner/opentuner_runner.py --json=./src/benchmarks/builtin_vectors/builtin_vectors.json
```

## Tentative script for running several tuners on the same problem
This currently requires opentuner being installed and the stop-after argument to opentuner being specified, 
irrespective of if opentuner is actually being executed. 
This due to the requirement of the argument parser being inherited from the opentuner parent parser.
```
python3 main.py --json=./src/benchmarks/builtin_vectors/builtin_vectors.json --stop-after=5
```
