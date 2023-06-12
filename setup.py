from setuptools import setup, find_packages
from typing import Dict, List, Tuple
from os import listdir
from pathlib import Path
import re

EXTRAS_ALL = "all"
TUNERS_PATH = "batbench/tuners"
README_PATH = "README.md"
REQ_FILE = "requirements.txt"

def read_requirements(file_path: Path) -> List[str]:
    return file_path.read_text().splitlines()

def add_tuner_requirements(tuners_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    extras_all = []
    extra_requirements: Dict[str, List[str]] = {}
    tuners_dir = Path(tuners_path)
    for tuner_path in tuners_dir.iterdir():
        if tuner_path.name.startswith("__"):
            continue
        tuner_requirements_file = tuner_path / REQ_FILE
        if tuner_requirements_file.exists():
            extra = tuner_path.name.split("_")[0]
            extra_requirements[extra] = read_requirements(tuner_requirements_file)
            extras_all += extra_requirements[extra]
        else:
            print(f"File {REQ_FILE} not found for tuner {tuner_path.name}")
    return extras_all, extra_requirements

def get_version(init_file_path: str) -> str:
    init_str = Path(init_file_path).read_text()
    match = re.search(r"^__version__ = ['\"](?P<version>[\w\.]+?)['\"]$", init_str, re.MULTILINE)
    if match:
        return match.group("version")
    else:
        raise ValueError("Could not find version in bat/__init__.py")

extras_all, extra_requirements = add_tuner_requirements(TUNERS_PATH)
extra_requirements[EXTRAS_ALL] = extras_all

long_description = Path(README_PATH).read_text()
version = get_version("batbench/__init__.py")

requirements = read_requirements(Path(REQ_FILE))

setup(
    name='batbench',
    version=version,
    packages=find_packages(),
    license='MIT',
    description='A GPU benchmark suite for autotuners',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jacob Odgård Tørring',
    author_email='jacob.torring@ntnu.no',
    url='https://github.com/NTNU-HPC-Lab/BAT', # add the url of your github repo
    install_requires=requirements, # add any dependencies your project would need
    extras_require=extra_requirements,
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
