from pathlib import Path
from setuptools import setup, find_packages

project_root = Path(__file__).parent.absolute()
reqs_file = project_root / "requirements.txt"
with open(reqs_file) as f:
    requirements = f.read().splitlines()

version_file = project_root / "abayes" / "version.txt"
with open(version_file) as f:
    version = f.read().strip()

if __name__ == "__main__":
    setup(
        name="abayes",
        version=version,
        desription="Bayesian AB testing with Stan and Python.",
        packages=find_packages(),
        install_requires=requirements,
        package_data={
            "abayes": [
                "templates/*.stan",
                "distributions/*.stan",
            ]
        },
        python_requires=">=3.10",
)
