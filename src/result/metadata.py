import jc
import subprocess
import os
import json
import xmltodict

from src.result.zenodo import Zenodo

class Metadata:
    @staticmethod
    def get_metadata():
        metadata = {}
        metadata["zenodo"] = Zenodo.get_zenodo_metadata()
        metadata["environment"] = Dataset.get_environment_metadata()
        metadata["hardware"] = Dataset.get_hardware_metadata()
        return metadata

    @staticmethod
    def get_environment_metadata():
        env_metadata = {}
        env_metadata["requirements"] = Dataset.save_requirements()
        env_metadata["lsb_release"] = Dataset.get_lsb_release()
        env_metadata["hostname"] = Dataset.get_hostname()
        return env_metadata

    @staticmethod
    def get_hardware_metadata():
        metadata = {}
        nvidia_smi_out = subprocess.run(["nvidia-smi", "--query", "-x"], capture_output=True)
        o = xmltodict.parse(nvidia_smi_out.stdout)
        del o["nvidia_smi_log"]["gpu"]["processes"]
        metadata["nvidia_query"] = o
        metadata["lshw"] = Dataset.get_lshw()
        lscpu_out = subprocess.run(["lscpu", "--json"], capture_output=True)
        metadata["lscpu"] = json.loads(lscpu_out.stdout)
        proc_out = subprocess.run(["cat", "/proc/meminfo"], capture_output=True)
        metadata["meminfo"] = jc.parse('proc_meminfo', proc_out.stdout.decode("utf-8"))
        metadata["lsblk"] = Dataset.get_lsblk()
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
    def get_lsblk():
        try:
            lsblk_out = subprocess.run(["lsblk", "-a"], capture_output=True)
            return jc.parse('lsblk', lsblk_out.stdout.decode("utf-8"))
        except Exception:
            return {}


    @staticmethod
    def get_lsb_release():
        lsb_release_out = subprocess.run(["lsb_release", "-a"], capture_output=True)
        r = {}
        for line in lsb_release_out.stdout.decode("utf-8").split("\n"):
            line_list = line.split("\t")
            if line_list[0] == "" or line_list[-1] == "":
                continue
            r[line_list[0]] = line_list[-1]
        return r

    @staticmethod
    def get_hostname():
        return subprocess.run(["hostname"], capture_output=True).stdout.decode("utf-8").strip()


    @staticmethod
    def get_lshw():
        try:
            lshw_out = subprocess.run(["lshw", "-json", "-sanitize"], capture_output=True)
            return json.loads(lshw_out.stdout)
        except Exception:
            return {}

