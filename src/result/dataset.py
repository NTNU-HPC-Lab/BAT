import subprocess
import json
import xmltodict
import pandas as pd

class Dataset:
    def __init__(self, spec):
        self.spec = spec
        self.search_settings = self.spec["SearchSettings"]
        self.ext = self.spec["General"]["OutputFormat"]
        self.benchmark_name = self.spec["General"]["BenchmarkName"]
        if self.ext not in ("JSON", "HDF5"):
            raise Exception("Invalid output format", self.ext)
        self.filename_root = "{}-{}-{}".format(self.search_settings["TunerName"], self.benchmark_name, self.search_settings["Trials"], self.ext)
        self.filename_data = "{}-results.{}".format(self.filename_root, self.ext.lower())
        self.filename_metadata = "{}-metadata.json".format(self.filename_root)
        self.data_path = "./results/{}".format(self.filename_data)
        self.metadata_path = "./results/{}".format(self.filename_metadata)
        self.results = []
        self.write_interval = 10

    def add_result(self, result):
        self.results.append(result)
        if len(self.results) % self.write_interval == 0:
            self.write_data()



    def write_metadata(self):
        metadata = {}
        nvidia_smi_out = subprocess.run(["nvidia-smi", "--query", "-x"], capture_output=True)
        o = xmltodict.parse(nvidia_smi_out.stdout)
        del o["nvidia_smi_log"]["gpu"]["processes"]
        metadata["nvidia_query"] = o

        lshw_out = subprocess.run(["lshw", "-json"], capture_output=True)
        metadata["lshw"] = json.loads(lshw_out.stdout)
        with open(self.metadata_path, 'w') as f:
            metadata_json = json.dumps(metadata, indent=4)
            f.write(metadata_json)

    def write_data(self):
        dump_results = {"results": []}
        for result in self.results:
            dump_results["results"].append(result.serialize())

        if self.ext == "JSON":
            with open(self.data_path, 'w') as f:
                json.dump(dump_results, f, indent=4)
        elif self.ext == "HDF5":
            df = pd.json_normalize(dump_results["results"])
            df.to_hdf(self.data_path, key="Results", mode="w", complevel=9)
        else:
            print("Unsupported file extention", self.ext)
