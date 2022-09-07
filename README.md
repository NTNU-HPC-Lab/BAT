<h1 align="center">
	<br>
	<br>
	<img width="360" src="./media/BAT-logo.svg" alt="BAT">
	<br>
	<br>
	<br>
</h1>

> A standardized benchmark suite for autotuners

BAT is a standardized benchmark suite for autotuners that is based on benchmarks from [SHOC](https://github.com/knutkirkhorn/shoc) and contains benchmarks for [CUDA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) programs. The benchmarks are for both whole programs and kernel-code. BAT will save all your `JSON` and `CSV` results to an own results directory after autotuning is completed. Then it will parse specified files and print out the best parameters found by the autotuner. The parameters and other benchmarking information will be printed out prettified in the terminal.

This benchmark suite will be useful for you if you're making your own autotuner and want to use the benchmarks for testing or would like to compare your autotuner to other known autotuners. BAT can also be used to check how a parameter's value changes for different architectures.

## Parameters
Parameters and search space for the algorithms can be seen in the `src` directory [here](./src).

## Prerequisites
- [Python 3](https://www.python.org/) 
- All the current tuners are python3-based tuners. C++ based tuners will be added soon.

## Set up readers
The Python-based T1-format reader requires the python library cupy to execute CUDA kernels from python.
This can be installed with
```
pip install -r src/readers/python/requirements.txt
```
The Python-based tuners that do not have native T1-support use this library to execute their configurations.


## Set up autotuners
Without using Docker, the following steps are required to download and install the autotuners:
### Python-based Tuners
- [Optuna](https://github.com/optuna/optuna)
    - Can be downloaded using `pip3 install -r src/tuners/optuna/requirements.txt`.
- [OpenTuner](https://github.com/ingunnsund/opentuner)
    - Can be downloaded using `pip3 install -r src/tuners/opentuner/requirements.txt`.
- [Kernel Tuner](https://github.com/benvanwerkhoven/kernel_tuner)
    - Can be downloaded using `pip3 install -r src/tuners/kerneltuner/requirements.txt`.
- [SMAC3](https://github.com/automl/SMAC3)
    - Can be downloaded using `pip3 install -r src/tuners/SMAC3/requirements.txt`.
- [HyperOpt](https://github.com/hyperopt/hyperopt)
    - Can be downloaded using `pip3 install -r src/tuners/hyperopt/requirements.txt`.
### C++-based tuners
- [CLTune](https://github.com/ingunnsund/CLTune)
    - Need to set the environment variable `KTT_PATH=/path/to/KTT` for using the benchmarks.
- [KTT](https://github.com/Fillo7/KTT)
    - Need to set the environment variable `CLTUNE_PATH=/path/to/CLTune` for using the benchmarks.
- [ATF](https://gitlab.com/mdh-project/atf)
    - Need to set the environment variable `ATF_PATH=/path/to/ATF` for using the benchmarks.

## Example run using the builtin_vectors benchmark and the Optuna tuner.
```
python3 main.py --tuner="optuna" --json=./src/benchmarks/builtin_vectors/builtin_vectors.json --trials=10
```

## Command-line arguments
### `--json [path]`, `-b [path]`
Default: `none`

Path to json for the benchmark to run. Example: `./src/benchmarks/builtin_vectors/builtin_vectors-CAFF.json`.

### `--tuner [name]`, `-a [name]`
Default: `none`

Tuner to run benchmarks for. Example: `optuna`.


## Citation
Use the following citation when publishing work using BAT.
```
@article{sund_bat_2021,
	title = {{BAT}: A Benchmark suite for {AutoTuners}},
	rights = {Copyright (c) 2021},
	issn = {1892-0721},
	url = {https://ojs.bibsys.no/index.php/NIK/article/view/915},
	pages = {44--57},
	number = {1},
	journaltitle = {Norsk {IKT}-konferanse for forskning og utdanning},
	author = {Sund, Ingunn and Kirkhorn, Knut A. and Tørring, Jacob O. and Elster, Anne. C},
	urldate = {2021-12-10},
	date = {2021-11-14},
	langid = {english},
	note = {Number: 1},
}
```
