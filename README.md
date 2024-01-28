# rrt-rs
Rust library containing Rapidly-exploring Random Tree variants for trajectory planning, wrapped for use with Python. Uses [PyO3](https://pyo3.rs/v0.19.2/) as interface, and [maturin](https://github.com/PyO3/maturin) for building the library as a Python package.

[![platform](https://img.shields.io/badge/platform-linux-lightgrey)]()
[![python version](https://img.shields.io/badge/python-3.10-blue)]()
[![python version](https://img.shields.io/badge/python-3.11-blue)]()

## Citation
If you are using `RRTs` for ship trajectory planning or behavior generation in your work, please use the following citation:
```
@Article{Tengesdal2024csrrt,
  author  = {Trym Tengesdal and Tom A. Pedersen and Tor A. Johansen},
  journal = {Ocean Engineering},
  title   = {A Comparative Study of Rapidly-exploring Random Tree Algorithms Applied to Ship Trajectory Planning and Behavior Generation},
  year    = {2023},
  note    = {Submitted},
}
```


## Python Dependencies
- maturin
- matplotlib
- shapely
- numpy
- seacharts: https://github.com/trymte/seacharts
- colav_simulator: https://github.com/NTNU-Autoship-Internal/colav_simulator (for the examples)

## Installation and usage in Python

- 1: Install python dependencies.
- 2: Install rustup on your computer: https://www.rust-lang.org/tools/install
- 3: Go into root folder `rrt-rs/` and run `maturin develop`. The rust library is now built and installed as a Python package.
- 4: Run an example under `examples/` to check the installation.


## Future enhancements/todos

- Properly separate rust library functionality from the PyO3 wrapping framework, to allow for easy usage in Rust without considering Python.
- Use optimization-based steering.

