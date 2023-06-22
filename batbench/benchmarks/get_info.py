import os
import json

def main():
    benchmarks = []
    for folder_name in os.listdir("."):
        try:
            with open(f"./{folder_name}/{folder_name}-CAFF.json", 'r', encoding='utf-8') as file:
                benchmark = {}
                text = file.read()
                benchmark_json = json.loads(text)
                benchmark["name"] = benchmark_json["General"].get("BenchmarkName")
                tuning_params = benchmark_json["ConfigurationSpace"]["TuningParameters"]
                benchmark["dimensionality"] = len(tuning_params)
                cardinality = 1
                for param in tuning_params:
                    cardinality *= len(eval(str(param["Values"])))
                benchmark["cardinality"] = cardinality
                benchmarks.append(benchmark)
        except FileNotFoundError:
            pass

    for benchmark in sorted(benchmarks, key=lambda item: item["cardinality"], reverse=True):
        print(f"Benchmark: {benchmark['name']:15} Dimensionality: {benchmark['dimensionality']:3}\t Cardinality: {benchmark['cardinality']:,}")

if __name__ == "__main__":
    main()
