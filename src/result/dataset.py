import subprocess
import json
import xmltodict
import pandas as pd

class Dataset:
    def __init__(self, spec):
        self.spec = spec
        self.search_settings = self.spec["SearchSettings"]
        ext = self.spec["General"]["OutputFormat"].lower()
        if ext not in ("json", "hdf5"):
            raise Exception("Invalid output format", ext)
        self.filename = "{}-{}-results.{}".format(self.search_settings["TunerName"], self.search_settings["Trials"], ext)
        self.results = []

    def add_result(self, result):
        self.results.append(result)

    @staticmethod
    def write_metadata():
        metadata = {}
        nvidia_smi_out = subprocess.run(["nvidia-smi", "--query", "-x"], capture_output=True)
        o = xmltodict.parse(nvidia_smi_out.stdout)
        del o["nvidia_smi_log"]["gpu"]["processes"]
        metadata["nvidia_query"] = o

        lshw_out = subprocess.run(["lshw", "-json"], capture_output=True)
        metadata["lshw"] = json.loads(lshw_out.stdout)
        with open('./results/optuna/{}'.format('metadata.json'), 'w') as f:
            metadata_json = json.dumps(metadata, indent=4)
            f.write(metadata_json)

    def write_data(self):
        dump_results = {"results": []}
        for result in self.results:
            dump_results["results"].append(result.serialize())

        ext = self.filename.split(".")[-1]
        if ext == "json":
            with open("./results/optuna/{}".format(self.filename), 'a') as f:
                json.dump(dump_results, f, indent=4)
        elif ext == "hdf5":
            df = pd.json_normalize(dump_results["results"])
            print(df)
            df.to_hdf(self.filename, key="Results", mode="w", complevel=9)
        else:
            print("Unsupported file extention", ext)

if __name__ == "__main__":
    Dataset.write_metadata()
