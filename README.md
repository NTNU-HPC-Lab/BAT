<h1 align="center">
	<br>
	<br>
	<img width="360" src="./media/BAT-logo.svg" alt="BAT">
	<br>
	<br>
	<br>
</h1>

## Overview of benchmark compatibility with tuners
|           | GEMM | Nbody | DeDisp | ExpDist | PnPoly | Convolution | Hotspot | MD5Hash | TRIAD |
|:---------:|:----:|:-----:|:------:|:-------:|:------:|:-----------:|:-------:|:-------:|:-----:|
| Opentuner |  ✅  |   ✅  |   ✅   |   ❌   |   ✅   |     ✅      |   ✅    |   ✅    |   ✅   |
| Kerneltuner | ✅  |   ❌  |   ✅   |   ✅   |   ✅   |     ✅      |   ✅    |   ✅    |   ✅   |
| Mintuner  |  ✅  |   ✅  |   ✅   |   ✅   |   ✅   |     ✅      |   ✅    |   ✅    |   ✅   |
| Optuna    |  ✅  |   ✅  |   ✅   |   ✅   |   ✅   |     ✅      |   ✅    |   ✅    |   ✅   |


> A GPU benchmark suite for autotuners

BAT is a standardized benchmark suite for autotuners that is based on benchmarks from [SHOC](https://github.com/vetter/shoc), Rodinia, [KTT](https://github.com/HiPerCoRe/KTT) and benchmarks from the Netherlands eScience Center. BAT will save all your `json` results after autotuning is completed. Then it will print out the best parameters found by the autotuner.

This benchmark suite will be useful for you if you're making your own autotuner and want to use the benchmarks for testing or would like to compare your autotuner to other known autotuners. BAT can also be used to check how a parameter's value changes for different architectures.

## Prerequisites
- [Python 3](https://www.python.org/) 
- All the current tuners are python3-based tuners. C++ based tuners will be added soon.

## Set up python dependencies
The requirements can be installed with
```
pip install -r requirements.txt
```

## Set up autotuners
The following steps are required to download and install the autotuners:
### Python-based Tuners
- [Optuna](https://github.com/optuna/optuna)
    - Can be downloaded using `pip3 install -r bat/tuners/optuna_runner/requirements.txt`.
- [OpenTuner](https://github.com/ingunnsund/opentuner)
    - Can be downloaded using `pip3 install -r bat/tuners/opentuner_runner/requirements.txt`.
- [Kernel Tuner](https://github.com/benvanwerkhoven/kernel_tuner)
    - Can be downloaded using `pip3 install -r bat/tuners/kerneltuner_runner/requirements.txt`.
- [SMAC3](https://github.com/automl/SMAC3)
    - Can be downloaded using `pip3 install -r bat/tuners/smac3_runner/requirements.txt`.
### C++-based tuners
Coming back soon.

## Example run using the GEMM benchmark and Mintuner.
```
python3 main.py --tuner mintuner --benchmark GEMM --trials 10
```

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
