![build](https://github.com/cmgoold/miniab/actions/workflows/test.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# miniab

`miniab` is a small Python package for performing Bayesian AB testing.
Computations are run using Stan via `cmdstanpy`, and Jinja2 is used
in the backend to construct the Stan model files.
