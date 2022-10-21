import subprocess
import json
import hashlib
import shutil
import os
import copy
from pathlib import Path
import xmltodict
import pandas as pd

from src.manager.util import get_spec, get_kernel_path
from src.result.zenodo import Zenodo
from src.result.result import Result

class Dataset:
    def __init__(self, spec_path):
        self.files = []
        self.results = []
        self.write_interval = 10

        self.spec = get_spec(spec_path)
        self.metadata = Dataset.get_metadata()
        self.spec_path = spec_path
        self.kernel_path = get_kernel_path(self.spec)

        self.ext = self.spec["General"]["OutputFormat"]
        self.benchmark_name = self.spec["General"]["BenchmarkName"]
        self.best_result = Result(self.spec)

        if self.ext not in ("JSON", "HDF5"):
            raise Exception("Invalid output format", self.ext)

        self.create_dataset_folder()

        self.copy_file("spec.json", self.spec_path)
        self.copy_file("search-spec.json", "./search-settings.json")
        self.create_source_folder()
        self.write_metadata(self.metadata)

        self.input_zip = "input-data.zip"
        self.results_setup()


    def zip_folders(self, files):
        # Zipping of folders
        for f in files:
            f_split = f.split(".")
            if f_split[-1] == 'zip':
                name = ''.join(f_split[:-1])
                shutil.make_archive(f"{self.path}/{name}", 'zip', f"{self.path}/{name}")

    def results_setup(self):
        # Generate name and paths for output files
        self.results_path = f"{self.path}/results"
        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        self.result_id = sum([1 if element.is_file() else 0 for element in Path(self.results_path).iterdir()]) + 1
        self.results_filename = f"{self.result_id}.{self.ext.lower()}"
        self.output_results_path = f"{self.results_path}/{self.results_filename}"
        results_zip = "results.zip"
        self.files.append(results_zip)

    def copy_file(self, filename, filepath):
        shutil.copyfile(filepath, f"{self.path}/{filename}")
        self.files.append(filename)

    def create_source_folder(self):
        self.source_zip = "source.zip"
        Path(f"{self.path}/source").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.kernel_path, f"{self.path}/source/kernel.cu")
        self.files.append(self.source_zip)

    def create_dataset_folder(self):
        # Create dataset folder
        self.root_path = "./results"
        hash_set = copy.deepcopy(self.spec)
        hash_set.update(self.metadata)
        del hash_set["General"]
        del hash_set["zenodo"]

        lshw = hash_set["environment"]["lshw"]
        hostname = lshw[0]["id"] if isinstance(lshw, list) else lshw["id"]

        del hash_set["environment"]["lshw"]
        hash_set["environment"]["hostname"] = hostname
        # del hash_set["environment"]["lshw"][0]["children"][0]["children"][1]["size"] TODO: This is not consistent across systems
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["timestamp"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["fan_speed"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["performance_state"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["fb_memory_usage"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["bar1_memory_usage"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["utilization"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["temperature"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["power_readings"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["clocks"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["applications_clocks"]
        del hash_set["environment"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["default_applications_clocks"]

        self.hash = hashlib.md5(json.dumps(hash_set,
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        self.path = f"{self.root_path}/{self.hash}"
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def update_best(self):
        for result in self.results:
            self.best_result = result.pickBest(self.best_result)

    def add_result(self, result):
        self.results.append(result)
        self.best_result = result.pickBest(self.best_result)

        if len(self.results) % self.write_interval == 0:
            self.write_data()

    def write_metadata(self, metadata):
        metadata_filename = "metadata.json"
        with open(f"{self.path}/{metadata_filename}", 'w') as f:
            f.write(json.dumps(metadata, indent=4))
        self.files.append(metadata_filename)

    @staticmethod
    def get_metadata():
        metadata = {}
        metadata["zenodo"] = Zenodo.get_zenodo_metadata()
        metadata["environment"] = Dataset.get_environment_metadata()
        return metadata

    @staticmethod
    def save_requirements():
        requirements_path = f"requirements-temp.txt"
        subprocess.call(['sh', './update-dependencies.sh', requirements_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(requirements_path, 'r') as f:
            requirements_list = [line.strip() for line in f.readlines()]
        os.remove(requirements_path)
        return requirements_list

    @staticmethod
    def get_hardware_metadata(metadata):
        nvidia_smi_out = subprocess.run(["nvidia-smi", "--query", "-x"], capture_output=True)
        o = xmltodict.parse(nvidia_smi_out.stdout)
        del o["nvidia_smi_log"]["gpu"]["processes"]
        metadata["nvidia_query"] = o
        lshw_out = subprocess.run(["lshw", "-json"], capture_output=True)
        metadata["lshw"] = json.loads(lshw_out.stdout)
        return metadata

    @staticmethod
    def get_environment_metadata():
        env_metadata = {}
        env_metadata["requirements"] = Dataset.save_requirements()
        env_metadata = Dataset.get_hardware_metadata(env_metadata)
        return env_metadata

    def write_data(self):
        dump_results = {"results": []}
        for result in self.results:
            dump_results["results"].append(result.serialize())

        p = self.output_results_path
        if self.ext == "JSON":
            with open(p, 'w') as f:
                json.dump(dump_results, f, indent=4)
        elif self.ext == "HDF5":
            df = pd.json_normalize(dump_results["results"])
            df.to_hdf(p, key="Results", mode="w", complevel=9)
        else:
            print("Unsupported file extention", self.ext)
