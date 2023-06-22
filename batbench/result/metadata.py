import subprocess
import json
import jc
import xmltodict

from batbench.result.zenodo import Zenodo

class Metadata:
    @staticmethod
    def get_metadata():
        metadata = {}
        metadata["zenodo"] = Zenodo.get_zenodo_metadata()
        metadata["environment"] = Metadata.get_environment_metadata()
        metadata["hardware"] = Metadata.get_hardware_metadata()
        return metadata

    @staticmethod
    def get_environment_metadata():
        env_metadata = {}
        env_metadata["requirements"] = Metadata.save_requirements()
        env_metadata["lsb_release"] = Metadata.get_lsb_release()
        env_metadata["hostname"] = Metadata.get_hostname()
        return env_metadata

    @staticmethod
    def get_hardware_metadata():
        metadata = {}
        nvidia_smi_out = subprocess.run(["nvidia-smi", "--query", "-x"],
                                        check=True, capture_output=True)
        nvidia_smi = xmltodict.parse(nvidia_smi_out.stdout)
        del nvidia_smi["nvidia_smi_log"]["gpu"]["processes"]
        metadata["nvidia_query"] = nvidia_smi
        metadata["lshw"] = Metadata.get_lshw()
        lscpu_out = subprocess.run(["lscpu", "--json"], check=True, capture_output=True)
        metadata["lscpu"] = json.loads(lscpu_out.stdout)
        proc_out = subprocess.run(["cat", "/proc/meminfo"], check=True, capture_output=True)
        metadata["meminfo"] = jc.parse('proc_meminfo', proc_out.stdout.decode("utf-8"))
        metadata["lsblk"] = Metadata.get_lsblk()
        return metadata

    @staticmethod
    def save_requirements():
        requirements_path = "requirements.txt"
        with open(requirements_path, 'r', encoding='utf-8') as file:
            requirements_list = [line.strip() for line in file.readlines()]
        return requirements_list

    @staticmethod
    def get_lsblk():
        try:
            lsblk_out = subprocess.run(["lsblk", "-a"], check=True, capture_output=True)
            return jc.parse('lsblk', lsblk_out.stdout.decode("utf-8"))
        except subprocess.CalledProcessError:
            return {}


    @staticmethod
    def get_lsb_release():
        lsb_release_out = subprocess.run(["lsb_release", "-a"],
                                         check=True, capture_output=True)
        lsb_json = {}
        for line in lsb_release_out.stdout.decode("utf-8").split("\n"):
            line_list = line.split("\t")
            if line_list[0] == "" or line_list[-1] == "":
                continue
            lsb_json[line_list[0]] = line_list[-1]
        return lsb_json

    @staticmethod
    def get_hostname():
        return subprocess.run(["hostname"], capture_output=True,
                              check=True).stdout.decode("utf-8").strip()


    @staticmethod
    def get_lshw():
        try:
            lshw_out = subprocess.run(["lshw", "-json", "-sanitize"],
                                      capture_output=True, check=True)
            return json.loads(lshw_out.stdout)
        except subprocess.CalledProcessError:
            return {}
