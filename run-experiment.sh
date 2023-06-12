#!/bin/bash
benchmarks=("pnpoly" "GEMM" "nbody" "convolution" "hotspot")
tuners=("opentuner" "kerneltuner" "mintuner" "optuna" "smac")
trials=12

for benchmark in "${benchmarks[@]}"
do
  for tuner in "${tuners[@]}"
  do
    python3 main.py --tuner $tuner --benchmark $benchmark --trials $trials
    folder_name="experiment_results/results_${benchmark}_${tuner}"
    mkdir -p $folder_name
    mv results/* experiment_results/$folder_name/
    if [ "$tuner" == "opentuner" ]; then
      rm -rf opentuner.* 2> /dev/null
    fi
    if [ "$tuner" == "smac" ]; then
      rm -rf smac3-output_* 2> /dev/null
    fi
  done
done
