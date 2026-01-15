# setup.py
from setuptools import setup, find_packages
from pathlib import Path

# Read requirements.txt
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path, encoding="utf-8") as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith("#")]

setup(
    name="roster_lib",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements(),
)
