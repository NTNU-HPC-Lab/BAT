from pathlib import Path
from typing import Dict, List
import re
from setuptools import setup, find_packages

EXTRAS_ALL = "all"
EXTRAS_TUNER = "tuners"
EXTRAS_BACKEND = "backends"

BACKENDS_PATH = "batbench/backends"
TUNERS_PATH = "batbench/tuners"
README_PATH = "README.md"
REQ_FILE = "requirements.txt"

def read_requirements(file_path: Path) -> List[str]:
    return file_path.read_text().splitlines()

def add_requirements(path: str, extras_type: str, extra_requirements: Dict[str, List[str]], 
                     name_extract_func=lambda name: name) -> List[str]:
    extras_all = []
    dir_path = Path(path)
    for item_path in dir_path.iterdir():
        if item_path.name.startswith("__"):
            continue
        requirements_file = item_path / REQ_FILE
        if requirements_file.exists():
            extra = name_extract_func(item_path.name)
            extra_requirements[extra] = read_requirements(requirements_file)
            extras_all += extra_requirements[extra]
        else:
            print(f"File {REQ_FILE} not found for {extras_type} {item_path.name}")
    return extras_all

def get_version(init_file_path: str) -> str:
    with open(init_file_path, 'r', encoding='utf-8') as file:
        init_str = file.read()
    match = re.search(r"^__version__ = ['\"](?P<version>[\w\.]+?)['\"]$", init_str, re.MULTILINE)
    if match:
        return match.group("version")
    raise ValueError("Could not find version in bat/__init__.py")


def get_extra_requirements():
    extra_requirements: Dict[str, List[str]] = {}
    extra_requirements[EXTRAS_TUNER] = add_requirements(
        TUNERS_PATH, EXTRAS_TUNER, extra_requirements, lambda name: name.split("_")[0])
    extra_requirements[EXTRAS_BACKEND] = add_requirements(
        BACKENDS_PATH, EXTRAS_BACKEND, extra_requirements)
    extra_requirements[EXTRAS_ALL] = extra_requirements[EXTRAS_BACKEND] + extra_requirements[EXTRAS_TUNER]
    return extra_requirements

long_description = Path(README_PATH).read_text(encoding='utf-8')
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
    url='https://github.com/NTNU-HPC-Lab/BAT',
    install_requires=requirements,
    extras_require=get_extra_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
