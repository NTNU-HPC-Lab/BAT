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
from src.result.metdata import Metadata
from src.result.result import Result

class Dataset:
    def __init__(self, spec):
        self.files = []
        self.cache_df = pd.DataFrame({})
        self.write_interval = 10

        self.spec = spec
        self.metadata = Metadata.get_metadata()
        self.metadata["spec"] = self.spec
        self.validate_schema(self.metadata)

        self.kernel_path = get_kernel_path(self.spec)

        self.ext = self.spec["General"]["OutputFormat"]
        self.benchmark_name = self.spec["General"]["BenchmarkName"]

        if self.ext not in ("JSON", "HDF5", "CSV"):
            raise Exception("Invalid output format", self.ext)

        self.create_dataset_folder()

        self.create_source_folder()
        self.write_metadata(self.metadata)

        self.input_zip = "input-data.zip"
        self.results_setup()

    def validate_schema(self, metadata):
        from jsonschema import validate
        with open('schemas/TuningSchema/metadata-schema.json', 'r') as f:
            schema = json.loads(f.read())
        validate(instance=metadata, schema=schema)

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
        self.cache_results_path = f"{self.results_path}/{self.result_id}.hdf"
        self.output_results_path = f"{self.results_path}/{self.results_filename}"
        results_zip = "results.zip"
        self.files.append(results_zip)

    def copy_file(self, filename, filepath):
        shutil.copyfile(filepath, f"{self.path}/{filename}")
        self.files.append(filename)

    def copy_and_delete_file(self, filename, filepath):
        self.copy_file(filename, filepath)
        os.remove(filepath)



    def create_source_folder(self):
        self.source_zip = "source.zip"
        Path(f"{self.path}/source").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.kernel_path, f"{self.path}/source/kernel.cu")
        self.files.append(self.source_zip)

    def create_dataset_folder(self):
        # Create dataset folder
        self.root_path = "./results"
        hash_set = copy.deepcopy(self.spec)
        copy_metadata = copy.deepcopy(self.metadata)
        hash_set.update(copy_metadata)
        del hash_set["General"]
        del hash_set["zenodo"]

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

        json_hash_set = json.dumps(hash_set,
            ensure_ascii=False,
            sort_keys=True,
            indent=4,
            #indent=None,
            separators=(',', ':'),
        )


        self.hash = hashlib.md5(json_hash_set.encode('utf-8')).hexdigest()
        #with open(f'test-{self.hash}.json', 'w') as f:
        #    f.write(json_hash_set)
        self.path = f"{self.root_path}/{self.hash}"
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def delete_files(self):
        shutil.rmtree(self.path)

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

    def write_metadata(self, metadata):
        metadata_filename = "metadata.json"
        with open(f"{self.path}/{metadata_filename}", 'w') as f:
            f.write(json.dumps(metadata, indent=4))
        self.files.append(metadata_filename)



    def flatten_df(self, df):
        df_json = df.to_json(orient="records")
        evaled_json = eval(df_json)
        df_flat = pd.json_normalize(evaled_json)
        #df_flat = df_flat.astype({"validity": "str"})
        df_flat = df_flat.astype({"times.runtimes": "str"})
        #df_flat = df_flat.astype({"times.compile_time": "float64"})
        #df_flat = df_flat.astype({"times.arg_time": "float64"})
        #df_flat = df_flat.astype({"times.total_time": "float64"})
        #df_flat = df_flat.astype({"times.algorithm_time": "float64"})
        #df_flat = df_flat.astype({"times.framework_time": "float64"})
        #df_concat = pd.concat([df1, df2])
        #df_result = df.join(df_flat)
        #return df_result
        return df_flat


    def write_data(self):
        #if self.ext == "CSV":
        #   self.cache_df.to_csv(self.output_results_path, mode='a', header=not os.path.exists(self.output_results_path))
        #elif self.ext == "HDF5":
        df = self.cache_df.reset_index()
        df = self.flatten_df(df)
        #print(df)

        df.to_hdf(self.cache_results_path, key="Results", mode="a", complevel=9, append=True, min_itemsize={"times.runtimes": 200})
        #else:
            #print("Unsupported file extention", self.ext)


    def final_write_data(self, df=None):
        if len(self.cache_df):
            self.write_data()
        df_iter = df if df is not None else pd.read_hdf(self.cache_results_path, "Results")
        df_iter.reset_index(drop=True, inplace=True)
        print(df_iter)
        if self.ext == "CSV":
            df_iter.to_csv(self.output_results_path, mode='w')
        elif self.ext == "JSON":
            df_iter = Dataset.to_formatted_df(df_iter)
            df_iter.to_json(self.output_results_path, orient="records", indent=4)
        elif self.ext == "HDF5":
            return
        else:
            print("Unsupported file extention", self.ext)

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

