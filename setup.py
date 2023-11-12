from pathlib import Path
from setuptools import setup, find_packages

project_root = Path(__file__).parent.absolute()
reqs_file = project_root / "requirements.txt"
with open(reqs_file) as f:
    requirements = f.read().splitlines()

if __name__ == "__main__":
    setup(
        name="abayestest",
        version="0.0.1",
        desription="Bayesian AB testing with Stan and Python.",
        packages=find_packages(),
        install_requires=requirements,
        package_data={
            "abayestest": [
                "templates/*.stan",
                "templates/distributions/*.stan",
                "templates/chunks/*.stan",
            ]
        },
        python_requires=">=3.10",
)
