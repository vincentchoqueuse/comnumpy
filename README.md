# Comnumpy : A Python Library for Communication System Prototyping and Simulation

[![Tests](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/tests.yml/badge.svg)](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/tests.yml)
[![Docs](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/docs.yml/badge.svg)](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/docs.yml)

A python library containing several Digital Signal Processing (DSP) algorithms for communication systems, including MIMO, OFDM, and optical fiber simulations.

`comnumpy` is made for **researchers**, **engineers**, and **students** in the field of **digital communications**. It’s ideal for anyone who wants to **simulate and analyze communication systems** without reinventing the wheel.

## Why choose `comnumpy`?

* 🧩 **Modular design**: Build and customize your own communication chains easily.
* ⚡ **Lightweight and efficient**: Around 400 KB of clean, well-structured code, without unnecessary dependencies.
* 📚 **Easy to use**: Clear API, comprehensive documentation, and ready-to-run examples.
* 🤝 **Open to contributions**: Developers are encouraged to add new submodules and extend the core capabilities.

## Prerequisites

All you need is a standard Python >3.11 setup with `numpy` and `scipy`. No need for bulky or domain-specific packages — `comnumpy` is ready to go with minimal setup.


## 📖 Documentation

The full documentation is available at:

👉 [https://vincentchoqueuse.github.io/comnumpy/](https://vincentchoqueuse.github.io/comnumpy/)

It includes:

* Quickstart tutorials
* API reference
* Examples for common use cases
* Developer guide for contributing new modules

## Getting Started

To use the package, you need to install it first. You can install directly from GitHub using pip:

```bash
pip install git+https://github.com/vincentchoqueuse/comnumpy.git
```

## Features

* MIMO signal processing algorithms
* OFDM modulation and demodulation
* Optical fiber link simulation with nonlinear effects
* Digital back-propagation for nonlinear compensation
* Symbol Error Rate (SER) computation and visualization tools

## Requirements

* Python 3.11
* numpy
* matplotlib

(see `requirements.txt`)

## Usage

Check out the `examples/` folder for usage demos, including MIMO, OFDM, and optical communication simulations.

## Contributing

Feel free to open issues or submit pull requests to improve the library.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE License. See the LICENSE file for details.
