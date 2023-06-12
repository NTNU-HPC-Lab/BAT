import os
import json

def main():
    benchmarks = []
    for folder_name in os.listdir("."):
        try:
            with open("./{folder_name}/{folder_name}-CAFF.json".format(folder_name=folder_name), 'r') as f:
                benchmark = {}
                text = f.read()
                j = json.loads(text)
                benchmark["name"] = j["General"].get("BenchmarkName")
                tuning_params = j["ConfigurationSpace"]["TuningParameters"]
                benchmark["dimensionality"] = len(tuning_params)
                cardinality = 1
                for param in tuning_params:
                    cardinality *= len(eval(str(param["Values"])))
                benchmark["cardinality"] = cardinality
                benchmarks.append(benchmark)
        except:
            pass

    for benchmark in sorted(benchmarks, key=lambda item: item["cardinality"], reverse=True):
        print("Benchmark: {0:15} Dimensionality: {1:3}\t Cardinality: {2:,}".format(benchmark["name"], benchmark["dimensionality"], benchmark["cardinality"]))

if __name__ == "__main__":
    main()

