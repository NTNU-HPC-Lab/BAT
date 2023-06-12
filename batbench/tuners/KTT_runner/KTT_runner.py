import subprocess
import json
import os
import pandas as pd

from batbench.manager import Manager
from batbench.result import Result

class KTTRunner:
    def __init__(self):
        self.path_to_ktt = "./bat/tuners/KTT_runner"

    def setup(self):
        wd = os.getcwd()
        os.chdir(self.path_to_ktt)
        subprocess.call("./KTT_setup.sh")
        os.chdir(wd)

    def run(self, args):
        wd = os.getcwd()
        os.chdir(self.path_to_ktt)
        subprocess.run([f"./KTT/Build/x86_64_Release/KttTuningLauncher ../../../{args.json}"],
         shell=True, check=True)
        os.chdir(wd)


    def main(self, args):
        self.manager = Manager(args)
        self.setup()
        self.run(args)
        tuning_output_path = f"{self.path_to_ktt}/TuningOutput.json"
        self.manager.dataset.df = self.convert_to_df(tuning_output_path)
        self.manager.dataset.df.to_hdf(self.manager.dataset.cache_results_path, key="Results", mode="a", complevel=9)
        self.manager.dataset.final_write_data()
        df2 = self.manager.dataset.flatten_df(self.manager.dataset.df)
        min_index = df2['objective'].idxmin()
        best_row = df2.loc[min_index]
        #return self.manager.dataset.get_best()
        return best_row

    def convert_to_df(self, path):
        result_list = []
        with open(path, 'r') as f:
            j = json.loads(f.read())

        for result in j["Results"]:
            conf = {}
            for param in result["Configuration"]:
                conf[param["Name"]] = param["Value"]

            runtimes = [ result["ComputationResults"][0]["Duration"]  / 1000000]
            objective = float(sum(runtimes)) / float(len(runtimes))

            res = Result(config=conf, objective=objective, runtimes=runtimes)

            res.arg_time = result["DataMovementOverhead"] / 1000000
            res.algorithm_time = result["SearcherOverhead"] / 1000000
            # res.validation_time = result["ValidationOverhead"]
            res.total_time = ( result["TotalDuration"] + result["TotalOverhead"] ) / 1000000
            res.compile_time = (result["TotalOverhead"] - res.arg_time - res.algorithm_time) / 1000000
            res.framework_time = res.total_time - res.compile_time - objective
            result_list.append(res)
        return pd.DataFrame([result.serialize() for result in result_list])




def main():
    args = "benchmarks/GEMM/GEMM-CAFF.json"
    KTTRunner().main(args)


if __name__ == "__main__":
    main()


