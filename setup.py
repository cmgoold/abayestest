from pathlib import Path
from setuptools import setup, find_packages

project_root = Path(__file__).parent.absolute()
reqs_file = project_root / "requirements.txt"
with open(reqs_file) as f:
    requirements = f.read().splitlines()

version_file = project_root / "miniab" / "version.txt"
with open(version_file) as f:
    version = f.read().strip()

if __name__ == "__main__":
    setup(
        name="miniab",
        version=version,
        desription="Bayesian AB testing with Stan and Python.",
        packages=find_packages(),
        install_requires=requirements,
        python_requires=">=3.10",
)
