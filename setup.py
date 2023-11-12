from pathlib import Path
from setuptools import setup, find_packages

dir = Path(__file__).parent
long_description = (dir / "README.md").read_text()

project_root = Path(__file__).parent.absolute()
reqs_file = project_root / "requirements.txt"
with open(reqs_file) as f:
    requirements = f.read().splitlines()

if __name__ == "__main__":
    setup(
        name="abayestest",
        version="0.0.4",
        desription="Bayesian AB testing with Stan and Python.",
        long_description=long_description,
        long_description_content_type="text/markdown",
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
