import subprocess
import json
import hashlib
import shutil
import os
from pathlib import Path
import xmltodict
import pandas as pd

from src.manager.util import get_spec, get_kernel_path
from src.result.zenodo import Zenodo

class Dataset:
    def __init__(self, spec_path):
        self.files = []
        self.results = []
        self.write_interval = 10

        self.spec = get_spec(spec_path)
        self.spec_path = spec_path
        self.kernel_path = get_kernel_path(self.spec)

        self.search_settings = self.spec["SearchSettings"]
        self.ext = self.spec["General"]["OutputFormat"]
        self.benchmark_name = self.spec["General"]["BenchmarkName"]

        if self.ext not in ("JSON", "HDF5"):
            raise Exception("Invalid output format", self.ext)

        self.create_dataset_folder()

        self.copy_spec()
        self.create_source_folder()
        self.write_metadata()

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

    def copy_spec(self):
        spec_filename = "spec.json"
        shutil.copyfile(self.spec_path, f"{self.path}/{spec_filename}")
        self.files.append(spec_filename)

    def create_source_folder(self):
        self.source_zip = "source.zip"
        Path(f"{self.path}/source").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.kernel_path, f"{self.path}/source/kernel.cu")
        self.files.append(self.source_zip)

    def create_dataset_folder(self):
        # Create dataset folder
        self.root_path = "./results"
        # TODO: Also include metadata json, so that the same config on different computers don't provide the same hash
        self.hash = hashlib.md5(json.dumps(self.spec,
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        self.path = f"{self.root_path}/{self.hash}"
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def write_metadata(self):
        metadata = {}
        metadata["zenodo"] = Zenodo.get_zenodo_metadata()
        metadata["environment"] = self.get_environment_metadata()
        metadata_filename = "metadata.json"
        with open(f"{self.path}/{metadata_filename}", 'w') as f:
            f.write(json.dumps(metadata, indent=4))
        self.files.append(metadata_filename)


    def save_requirements(self):
        requirements_path = f"{self.path}/requirements.txt"
        subprocess.call(['sh', './update-dependencies.sh', requirements_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(requirements_path, 'r') as f:
            requirements_list = [line.strip() for line in f.readlines()]
        os.remove(requirements_path)
        return requirements_list

    def add_result(self, result):
        self.results.append(result)
        if len(self.results) % self.write_interval == 0:
            self.write_data()

    def get_hardware_metadata(self, metadata):
        nvidia_smi_out = subprocess.run(["nvidia-smi", "--query", "-x"], capture_output=True)
        o = xmltodict.parse(nvidia_smi_out.stdout)
        del o["nvidia_smi_log"]["gpu"]["processes"]
        metadata["nvidia_query"] = o
        lshw_out = subprocess.run(["lshw", "-json"], capture_output=True)
        metadata["lshw"] = json.loads(lshw_out.stdout)
        return metadata

    def get_environment_metadata(self):
        env_metadata = {}
        env_metadata["requirements"] = self.save_requirements()
        env_metadata = self.get_hardware_metadata(env_metadata)
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
