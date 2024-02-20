import os
import copy
import json
import hashlib
import shutil
from pathlib import Path

import warnings
import pandas as pd

from jsonschema import validate
from tables import NaturalNameWarning

from batbench.util import SCHEMA_PATH
from batbench.result.metadata import Metadata

# PyTables doesn't like that the field times.runtimes has a dot in it's name.
# This warning only occured when adding it to the min_itemsize dictionary.
warnings.filterwarnings('ignore', category=NaturalNameWarning)


class Dataset:
    def __init__(self, experiment_settings, benchmark_name, objective, minimize):
        self.files = []
        self.cache_df = pd.DataFrame({})
        self.writes = 0
        self.write_interval = 10
        self.objective = objective
        self.minimize = minimize

        self.root_path = "./results"
        self.input_zip = "input-data.zip"
        self.metadata_filename = "metadata.json"
        self.results_zip = "results.zip"
        self.nvidia = False

        self.metadata = Metadata.get_metadata()
        #self.metadata["spec"] = self.spec
        #self.validate_schema(self.metadata)

        #self.kernel_path = get_kernel_path(self.spec)

        #self.output_format = self.spec["General"]["OutputFormat"].lower()
        self.benchmark_name = benchmark_name
        self.output_format = "csv"
        #self.benchmark_name = self.spec["General"]["BenchmarkName"]

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
        #shutil.copyfile(self.kernel_path, self.source_folder / "kernel.cu")
        self.files.append(self.source_zip)

    def create_results_folder(self):
        # Generate name and paths for output files
        self.results_path = self.dataset_folder / "results"
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.result_id = sum(1 for element in Path(self.results_path).iterdir()
                             if element.is_file()) + 1

        self.results_filename = f"{self.result_id}.{self.output_format.lower()}"
        #self.cache_results_path = self.results_path / f"{self.result_id}.hdf"
        self.cache_results_path = self.results_path / f"{self.result_id}.csv"
        self.output_results_path = self.results_path / self.results_filename

        self.files.append(self.results_zip)

    def create_dataset_folder(self):
        # Create dataset folder
        hash_set = self.get_hash_set()
        self.hash = self.calculate_hash(hash_set)

        self.dataset_folder = Path(self.root_path) / self.benchmark_name / self.hash
        self.dataset_folder.mkdir(parents=True, exist_ok=True)

    def get_hash_set(self):
        #hash_set = copy.deepcopy(self.spec)
        hash_set = {}
        copy_metadata = copy.deepcopy(self.metadata)
        hash_set.update(copy_metadata)

        try:
            del hash_set["General"]
            del hash_set["zenodo"]
            del hash_set["hardware"]["lshw"]
            del hash_set["hardware"]["lscpu"]
            del hash_set["hardware"]["meminfo"]
            del hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["timestamp"]
        except KeyError:
            pass
        if self.nvidia:
            gpu_log = hash_set["hardware"]["nvidia_query"]["nvidia_smi_log"]["gpu"]
            try:
                del gpu_log["fan_speed"]
                del gpu_log["clocks_throttle_reasons"]
                del gpu_log["performance_state"]
                del gpu_log["fb_memory_usage"]
                del gpu_log["bar1_memory_usage"]
                del gpu_log["utilization"]
                del gpu_log["temperature"]
                del gpu_log["power_readings"]
                del gpu_log["clocks"]
                del gpu_log["applications_clocks"]
                del gpu_log["default_applications_clocks"]
                del gpu_log["voltage"]
                del gpu_log["pci"]["rx_util"]
                del gpu_log["pci"]["tx_util"]
                del gpu_log["pci"]["pci_gpu_link_info"]["pcie_gen"]["current_link_gen"]
                del gpu_log["pci"]["pci_gpu_link_info"]["pcie_gen"]["device_current_link_gen"]
            except KeyError:
                pass
        return hash_set

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
        for file in files:
            f_split = file.split(".")
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
        #df = pd.read_hdf(self.cache_results_path, "Results")
        df = pd.read_csv(self.cache_results_path)
        df.reset_index(drop=True, inplace=True)
        best_index = df['objective'].idxmin() if self.minimize else df['objective'].idxmax()
        best_row = df.loc[best_index]
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
        with open(self.results_path / self.metadata_filename, 'w',
                  encoding='utf-8') as file:
            file.write(json.dumps(self.metadata, indent=4))
        self.files.append(self.metadata_filename)

    def write_data(self):
        df = self.cache_df.reset_index()
        df = self.flatten_df(df)
        df.to_csv(self.cache_results_path,
                  mode='a' if self.writes > 0 else 'w',
                  header=self.writes == 0,
                  index=False)
        self.writes += 1
        #df.to_hdf(self.cache_results_path, key="Results", mode="a",
        # complevel=9, append=True, min_itemsize={"times.runtimes": 200})

    def final_write_data(self, df=None):
        if len(self.cache_df):
            self.write_data()
        #df_iter = df if df is not None else pd.read_hdf(self.cache_results_path, "Results")
        df_iter = df if df is not None else pd.read_csv(self.cache_results_path)
        df_iter.reset_index(drop=True, inplace=True)
        print(df_iter, self.output_results_path)
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
        with open(f'{SCHEMA_PATH}/metadata-schema.json', 'r',
                  encoding='utf-8') as file:
            schema = json.loads(file.read())
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
        Given a dictionary (my_dict), a list of keys (key_arr), and a value (val), 
        this method sets the value at the path in the dictionary defined by the keys. 
        The keys in the list define a path in the dictionary where each key 
        (except the last one) corresponds to a nested dictionary. 
        The last key corresponds to the key where the value should be set.

        For example, given my_dict={}, key_arr=['a', 'b', 'c'], and val=10,
        the method modifies my_dict to be {'a': {'b': {'c': 10}}}.
        """
        # Start with the input dictionary
        current = my_dict
        # Enumerate through each key in the list
        for i, key in enumerate(key_arr):
            # Check if current key is not present in the current dictionary
            if key not in current:
                # If this is the last key in the list, set the value
                # Otherwise, create a new dictionary for this key
                current[key] = val if i==len(key_arr)-1 else {}
            else:
                # If the key is present but not associated with a dictionary,
                # raise an error as we can't add a sub-key to it
                if not isinstance(current.get(key), dict):
                    print("Given dictionary is not compatible with key structure requested")
                    raise ValueError("Dictionary key already occupied")

            # Move on to the next level of the dictionary
            current = current[key]

        # Return the modified input dictionary
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
