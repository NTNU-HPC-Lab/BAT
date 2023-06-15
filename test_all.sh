#!/bin/bash
#benchmarks=("GEMM")
benchmarks=("GEMM" "dedisp" "expdist" "pnpoly" "convolution" "hotspot" "MD5Hash" "TRIAD")
#benchmarks=("dedisp" "expdist" "pnpoly" "convolution" "hotspot" "MD5Hash" "TRIAD")
#tuners=("mintuner" "kerneltuner")
#tuners=("optuna" "mintuner" "kerneltuner" "smac" "opentuner")
tuners=("kerneltuner")
trials=25

for tuner in "${tuners[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        command="python3 -m batbench --tuner $tuner --benchmark $benchmark --trials $trials --cleanup=True"
        echo "$command"
        # Uncomment the line below to execute the command
        $command
    done
done

