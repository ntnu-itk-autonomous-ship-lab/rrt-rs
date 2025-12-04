# rrt-rs
Rust library containing multiple Rapidly-exploring Random Tree variants for trajectory planning and behavior generation that were compared in [A Comparative Study of Rapidly-exploring Random Tree Algorithms Applied to Ship Trajectory Planning and Behavior Generation](https://link.springer.com/article/10.1007/s10846-025-02222-7).

Wrapped for use with Python. Uses [PyO3](https://pyo3.rs/v0.19.2/) as interface, and [maturin](https://github.com/PyO3/maturin) for building the library as a Python package. Typically used with the [colav-simulator](https://github.com/NTNU-TTO/colav-simulator) framework for generating obstacle ship trajectories or as an own-ship trajectory planner.

[![platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)]()
[![python version](https://img.shields.io/badge/python-3.10-blue)]()
[![python version](https://img.shields.io/badge/python-3.11-blue)]()

## Citation
If you are using `RRTs` for ship trajectory planning or behavior generation in your work, please use the following citation:
```
@article{tengesdal2025comparative,
  title={A Comparative Study of Rapidly-exploring Random Tree Algorithms Applied to Ship Trajectory Planning and Behavior Generation},
  author={Tengesdal, Trym and Pedersen, Tom Arne and Johansen, Tor Arne},
  journal={Journal of Intelligent \& Robotic Systems},
  volume={111},
  number={1},
  pages={1--19},
  year={2025},
  publisher={Springer}
}
```

## Installation and usage in Python

Install rustup on your computer: (<https://www.rust-lang.org/tools/install>):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install `uv` (<https://docs.astral.sh/uv/getting-started/installation/>):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync --group test # To get colav-simulator aswell
uv run maturin develop --uv
```
The rust library is now built and installed as a Python package.

Run/debug an example under `examples/` to check the installation.


## Future enhancements/todos

- Properly separate the rust library functionality from the PyO3 wrapping framework, to allow for easy usage in Rust without considering Python.
- Use optimization-based steering.

