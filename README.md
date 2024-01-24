<h1 align="center">
	<br>
	<br>
	<img width="360" src="./media/BAT-logo.svg" alt="BAT">
	<br>
	<br>
	<br>
</h1>


> A GPU benchmark suite for autotuners

BAT is a standardized benchmark suite for autotuners that is based on benchmarks from [SHOC](https://github.com/vetter/shoc), Rodinia, [KTT](https://github.com/HiPerCoRe/KTT) and benchmarks from the Netherlands eScience Center. BAT will save all your `json` results after autotuning is completed. Then it will print out the best parameters found by the autotuner.

This benchmark suite will be useful for you if you're making your own autotuner and want to use the benchmarks for testing or would like to compare your autotuner to other known autotuners. BAT can also be used to check how a parameter's value changes for different architectures.

## Prerequisites
- [Python 3](https://www.python.org/) 
- CUDA
- All the current tuners are python3-based tuners. C++ based tuners will be added soon.

# Installation
Barebones install with a simple tuner
```
pip3 install batbench 
```

Full installation with all tuners
```
pip3 install batbench[all]
```

## Install specific autotuners
The following steps are required to install the autotuners:
### Python-based Tuners
- Mintuner
    - Simple autotuner included in the package
- [Optuna](https://github.com/optuna/optuna)
    - Can be installed using `pip3 install batbench[optuna]`.
- [OpenTuner](https://github.com/ingunnsund/opentuner)
    - Can be installed using `pip3 install batbench[opentuner]`.
- [Kernel Tuner](https://github.com/benvanwerkhoven/kernel_tuner)
    - Can be installed using `pip3 install batbench[kerneltuner]`.
- [SMAC3](https://github.com/automl/SMAC3)
    - Can be installed using `pip3 install batbench[smac3]`.

### C++-based tuners
Coming back soon.

### Benchmark compatibility with tuners
|           | GEMM | Nbody | DeDisp | ExpDist | PnPoly | Convolution | Hotspot | MD5Hash | TRIAD |
|:---------:|:----:|:-----:|:------:|:-------:|:------:|:-----------:|:-------:|:-------:|:-----:|
| Opentuner |  ✅  |   ✅  |   ✅   |   ✅   |   ✅   |     ✅      |   ✅    |   ✅    |   ✅   |
| Kerneltuner | ✅  |   ❌  |   ✅   |   ✅   |   ✅   |     ✅      |   ✅    |   ✅    |   ✅   |
| Mintuner  |  ✅  |   ✅  |   ✅   |   ✅   |   ✅   |     ✅      |   ✅    |   ✅    |   ✅   |
| Optuna    |  ✅  |   ✅  |   ✅   |   ✅   |   ✅   |     ✅      |   ✅    |   ✅    |   ✅   |
| SMAC3 |  ✅  |   ✅  |   ✅   |   ✅   |   ✅   |     ✅      |   ✅    |   ✅    |   ✅   |

# Usage
```
python3 -m batbench --tuner TUNER --benchmark BENCHMARK
```

# Citation
Use the following citation when publishing work using BAT.
```
@inproceedings{torring_towards_2023,
	title = {Towards a Benchmarking Suite for Kernel Tuners},
	url = {https://ieeexplore.ieee.org/abstract/document/10196663},
	doi = {10.1109/IPDPSW59300.2023.00124},
	eventtitle = {2023 {IEEE} International Parallel and Distributed Processing Symposium Workshops ({IPDPSW})},
	pages = {724--733},
	booktitle = {2023 {IEEE} International Parallel and Distributed Processing Symposium Workshops ({IPDPSW})},
	author = {Tørring, Jacob O. and van Werkhoven, Ben and Petrovč, Filip and Willemsen, Floris-Jan and Filipovič, Jiří and Elster, Anne C.},
	urldate = {2024-01-24},
	date = {2023-05},
	keywords = {Computer architecture, Graphics processing units, Benchmark testing, Manuals, benchmarking, autotuning, Tuners, Codes, Distributed processing},
}
```

Or this one if you are referring to or using the original version of BAT (1.0). 
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
