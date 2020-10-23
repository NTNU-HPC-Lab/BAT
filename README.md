<h1 align="center">
	<br>
	<br>
	<img width="360" src="./media/BAT-logo.svg" alt="BAT">
	<br>
	<br>
	<br>
</h1>

> A standardized benchmark suite for auto-tuners

## Prerequisites
- [Python 3](https://www.python.org/)

## Running benchmarks
```sh
# Run all benchmark for all auto-tuners
python3 main.py

# Run the `sort` benchmark for all auto-tuners
python3 main.py -b sort

# Run all benchmarks for auto-tuner `OpenTuner`
python3 main.py -a opentuner

# Run benchmark `scan` for auto-tuner `CLTune`
python3 main.py -b scan -a cltune
```

## Command-line arguments
### `--benchmark [name]`, `-b [name]`
Default: `none`

Benchmark to run. Example: `sort`. If no benchmark is selected, all benchmarks are ran for selected auto-tuner(s).

### `--auto-tuner [name]`, `-a [name]`
Default: `none`

Auto-tuner to run benchmarks for. Example: `ktt`. If no auto-tuner is selected, all auto-tuners are selected for benchmarking.

### `--verbose`, `-v`
Default: `false`

If all `stdout` and `stderr` should be printed out during building of the benchmark(s). By default it does not print out the information during the building.

### `--size [number]`, `-s [number]`
Default: `1`

Problem size for the data in the benchmarks. By default it uses a problem size of `1`. This is up to the specific auto-tuner to handle.

### `--technique [name]`, `-t [name]`
Default: `brute_force`

Tuning technique to use for benchmarking. If no technique is specified, the brute force technique is selected. This is up to the specific auto-tuner to handle.

## Add your own auto-tuner
It is easy to add new auto-tuner implementations for the benchmarks, just follow these steps:
1. Store your auto-tuner implementation of a benchmark inside a auto-tuner subdirectory in [tuning_examples](./tuning_examples). The path to the benchmark implementation should look similar to `./tuning_examples/kernel_tuner/sort/`.
2. Create a `config.json` file in the same directory as the auto-tuner with content similar to this:
```json
{
    "build": [
        "make clean",
        "make"
    ],
    "run": "./sort",
    "results": [
        "best-sort-results.json"
    ]
}
```
