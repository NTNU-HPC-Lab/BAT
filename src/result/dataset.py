import os
import copy
from pathlib import Path

import pandas as pd
import json
import hashlib
import shutil

import warnings
from tables import NaturalNameWarning

# PyTables doesn't like that the field times.runtimes has a dot in it's name. This warning only occured when adding it to the min_itemsize dictionary.
warnings.filterwarnings('ignore', category=NaturalNameWarning)

from src.manager.util import get_spec, get_kernel_path
from src.result.metadata import Metadata
from src.result.result import Result

class Dataset:
    def __init__(self, spec):
        self.files = []
        self.cache_df = pd.DataFrame({})
        self.write_interval = 10

        self.spec = spec
        self.root_path = "./results"
        self.input_zip = "input-data.zip"
        self.metadata_filename = "metadata.json"
        self.results_zip = "results.zip"

        self.metadata = Metadata.get_metadata()
        self.metadata["spec"] = self.spec
        self.validate_schema(self.metadata)

        self.kernel_path = get_kernel_path(self.spec)

        self.output_format = self.spec["General"]["OutputFormat"].lower()
        self.benchmark_name = self.spec["General"]["BenchmarkName"]

        if self.output_format not in ("json", "hdf5", "csv"):
            raise ValueError(f"Output format '{self.output_format}' is not recognized.")

        self.create_dataset_folder()
        self.create_source_folder()
        self.create_results_folder()
        self.write_metadata()
    
    
    def create_source_folder(self):
        self.source_zip = "source.zip"
        self.source_folder = self.dataset_folder / "source"
        self.source_folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.kernel_path, self.source_folder / "kernel.cu")
        self.files.append(self.source_zip)

    def create_results_folder(self):
        # Generate name and paths for output files
        self.results_path = self.dataset_folder / "results"
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.result_id = sum([1 if element.is_file() else 0 for element in Path(self.results_path).iterdir()]) + 1
        self.results_filename = f"{self.result_id}.{self.output_format.lower()}"
        self.cache_results_path = self.results_path / f"{self.result_id}.hdf"
        self.output_results_path = self.results_path / self.results_filename
        
        self.files.append(self.results_zip)

    def create_dataset_folder(self):
        # Create dataset folder
        hash_set = self.get_hash_set()
        self.hash = self.calculate_hash(hash_set)

        self.dataset_folder = Path(self.root_path) / self.benchmark_name / self.hash
        self.dataset_folder.mkdir(parents=True, exist_ok=True)

    def get_hash_set(self):
        hash_set = copy.deepcopy(self.spec)
        copy_metadata = copy.deepcopy(self.metadata)
        hash_set.update(copy_metadata)
        del hash_set["General"]
        del hash_set["zenodo"]
        try: 
            del hash_set["hardware"]["lshw"]
            del hash_set["hardware"]["lscpu"]
            del hash_set["hardware"]["meminfo"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["timestamp"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["fan_speed"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["clocks_throttle_reasons"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["performance_state"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["fb_memory_usage"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["bar1_memory_usage"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["utilization"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["temperature"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["power_readings"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["clocks"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["applications_clocks"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["default_applications_clocks"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["voltage"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["pci"]["rx_util"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["pci"]["tx_util"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["pci"]["pci_gpu_link_info"]["pcie_gen"]["current_link_gen"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]["pci"]["pci_gpu_link_info"]["pcie_gen"]["device_current_link_gen"]
        except:
            pass

    def calculate_hash(self, hash_set: dict) -> str:
        json_hash_set = json.dumps(hash_set,
            ensure_ascii=False,
            sort_keys=True,
            indent=4,
            separators=(',', ':'),
        )

        return hashlib.md5(json_hash_set.encode('utf-8')).hexdigest()


    def zip_folders(self, files):
        # Zipping of folders
        for f in files:
            f_split = f.split(".")
            if f_split[-1] == 'zip':
                name = ''.join(f_split[:-1])
                sub_folder = self.dataset_folder / name
                shutil.make_archive(sub_folder, 'zip', sub_folder)

    def copy_file(self, filename, filepath):
        shutil.copyfile(filepath, self.dataset_folder / filename)
        self.files.append(filename)

    def copy_and_delete_file(self, filename, filepath):
        self.copy_file(filename, filepath)
        os.remove(filepath)

    def delete_files(self):
        shutil.rmtree(self.dataset_folder)

    def get_best(self):
        df = pd.read_hdf(self.cache_results_path, "Results")
        df.reset_index(drop=True, inplace=True)
        min_index = df['objective'].idxmin()
        best_row = df.loc[min_index]
        return best_row

    def add_result(self, result):
        new_df = pd.DataFrame([result.serialize()])
        self.cache_df = pd.concat([self.cache_df.reset_index(), new_df]).set_index('config')
        if "index" in self.cache_df.columns:
            self.cache_df = self.cache_df.drop(columns="index")

        if len(self.cache_df.index) == self.write_interval:
            self.write_data()
            self.cache_df = pd.DataFrame({})

    def write_metadata(self):
        with open(self.results_path / self.metadata_filename, 'w') as f:
            f.write(json.dumps(self.metadata, indent=4))
        self.files.append(self.metadata_filename)

    def write_data(self):
        df = self.cache_df.reset_index()
        df = self.flatten_df(df)
        df.to_hdf(self.cache_results_path, key="Results", mode="a", complevel=9, append=True, min_itemsize={"times.runtimes": 200})

    def final_write_data(self, df=None):
        if len(self.cache_df):
            self.write_data()
        df_iter = df if df is not None else pd.read_hdf(self.cache_results_path, "Results")
        df_iter.reset_index(drop=True, inplace=True)
        print(df_iter)
        if self.output_format == "csv":
            df_iter.to_csv(self.output_results_path, mode='w')
        elif self.output_format == "json":
            df_iter = Dataset.to_formatted_df(df_iter)
            df_iter.to_json(self.output_results_path, orient="records", indent=4)
        elif self.output_format == "hdf5":
            return
        else:
            print("Unsupported file extention", self.output_format)


    @staticmethod
    def validate_schema(metadata):
        from jsonschema import validate
        with open('schemas/TuningSchema/metadata-schema.json', 'r') as f:
            schema = json.loads(f.read())
        validate(instance=metadata, schema=schema)


    @staticmethod
    def flatten_df(df):
        df_json = df.to_json(orient="records")
        evaled_json = eval(df_json)
        df_flat = pd.json_normalize(evaled_json)
        df_flat = df_flat.astype({"times.runtimes": "str"})
        return df_flat

    @staticmethod
    def set_for_keys(my_dict, key_arr, val):
        """
        Set val at path in my_dict defined by the string (or serializable object) array key_arr
        """
        current = my_dict
        for i in range(len(key_arr)):
            key = key_arr[i]
            if key not in current:
                if i==len(key_arr)-1:
                    current[key] = val
                else:
                    current[key] = {}
            else:
                if type(current[key]) is not dict:
                    print("Given dictionary is not compatible with key structure requested")
                    raise ValueError("Dictionary key already occupied")

            current = current[key]

        return my_dict

    @staticmethod
    def to_formatted_json(df, sep="."):
        result = []
        for _, row in df.iterrows():
            parsed_row = {}
            for idx, val in row.items():
                keys = idx.split(sep)
                parsed_row = Dataset.set_for_keys(parsed_row, keys, val)

            result.append(parsed_row)
        return result

    @staticmethod
    def to_formatted_df(df):
        return pd.DataFrame(Dataset.to_formatted_json(df))

