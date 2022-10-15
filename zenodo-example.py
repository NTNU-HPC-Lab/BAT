from src.result.zenodo import Zenodo, Dataset
import json
import os

def make_metadata():
    metadata = {}
    metadata["title"] = "BAT Submission"
    metadata["access_right"] = "open"
    metadata["upload_type"] = "dataset"
    metadata["communities"] = [{"identifier": "autotuning-benchmarking"}]
    metadata["description"] = "This is a submission for the BAT benchmark platform"
    creators = []
    creators.append({"name": "Doe, John", "affiliation": "Zenodo"})
    metadata["creators"] = creators
    return metadata

if __name__ == "__main__":
    data = { "metadata": make_metadata() }
    with open("metadata-zenodo.json", "w") as f:
        f.write(json.dumps(data, indent=4))
    #access_token = os.environ.get('access_token')
    #dataset = Dataset("benchmarks/GEMM/GEMM-CAFF.json")
    #z = Zenodo(access_token, dataset)
    #print(z.get_user_depositions())
    #z.new(data)
    #z.upload_files()
    #print(z.get_user_depositions())

