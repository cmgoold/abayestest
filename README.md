![build](https://github.com/cmgoold/miniab/actions/workflows/test.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# miniab

`miniab` is a lightweight Python package for performing Bayesian AB testing.
Computations are run using Stan via [`cmdstanpy`](
https://github.com/stan-dev/cmdstanpy
), and [`jinja2`](
https://github.com/pallets/jinja/
) is used
in the backend to construct the Stan model files.
