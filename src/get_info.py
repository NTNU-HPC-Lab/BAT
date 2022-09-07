import os
import json

for folder_name in os.listdir("src/benchmarks/"):
    try:
        with open("src/benchmarks/{folder_name}/{folder_name}-CAFF.json".format(folder_name=folder_name), 'r') as f:
            text = f.read()
            j = json.loads(text)
            name = j["general"].get("benchmarkName")
            cardinality = 1
            tuning_params = j["configurationSpace"]["tuningParameters"]
            dimensionality = len(tuning_params)
            for param in tuning_params:
                cardinality *= len(eval(str(param["values"])))
            print("Benchmark: {0:15} Dimensionality: {1:3}\t Cardinality: {2:,}".format(name, dimensionality, cardinality))
    except:
        pass

