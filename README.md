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
- [Python 3](https://www.python.org/) (Or [Docker](https://www.docker.com/), see section <a href="#within-a-docker-container">Within a Docker container</a>)

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

### Content of `config.json`
- `build`: A list of commands that will be ran before the `run` command. Note, it does not work correctly with `&&` between commands. This is because of a limitation in the package [subprocess](https://docs.python.org/3/library/subprocess.html) to run the commands in Python. A solution is therefore to split them in a list.
- `run`: The command to run the auto-tuning benchmark.
- `results`: A list of result files that contains the best parameters found in the auto-tuner benchmark. These will be printed out by BAT after the auto-tuning is completed.

## Within a Docker container
### Building
Here are some examples of how to build the different auto-tuner Docker images:
```sh
# Build OpenTuner Dockerfile
$ docker build -t bat-opentuner -f docker/opentuner.Dockerfile .

# Build Kernel Tuner Dockerfile
$ docker build -t bat-kernel_tuner -f docker/kernel_tuner.Dockerfile .

# Build CLTune Dockerfile
$ docker build -t bat-cltune -f docker/cltune.Dockerfile .

# Build KTT Dockerfile
$ docker build -t bat-ktt -f docker/ktt.Dockerfile .
```

### Running
Here are some examples of how to run the different auto-tuner Docker containers:
```sh
# Run the KTT container
$ docker run -ti --gpus all bat-ktt

# Example of running container detatched
$ docker run -d -ti --gpus all bat-ktt

# Open a shell into a detatched container
$ docker exec -it <container-id> sh

# After this the commands shown in the `Running benchmarks` section can be used
# Example:
$ main.py -b sort -a ktt -t mcmc -s 4
```
