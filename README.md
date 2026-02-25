# comnumpy

**A Python library for communication system prototyping and simulation.**

[![Tests](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/tests.yml/badge.svg)](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/tests.yml)
[![Docs](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/docs.yml/badge.svg)](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`comnumpy` provides Digital Signal Processing (DSP) algorithms for communication systems, covering MIMO, OFDM, and optical fiber simulations. It is designed for **researchers**, **engineers**, and **students** who want to simulate and analyze communication systems without implementing standard algorithms from scratch.

## Why comnumpy?

- **Modular design** — Build custom communication chains by combining reusable `Processor` blocks with `Sequential`, inspired by PyTorch’s `nn.Module` pattern.
- **Lightweight** — Around 400 KB of clean code. Only requires `numpy` and `scipy`.
- **Comprehensive** — Covers AWGN, OFDM, MIMO, and optical fiber channels with nonlinear propagation.
- **Well documented** — Tutorials with math, diagrams, and ready-to-run examples.

## Quick Example

```python
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.channels import AWGN
from comnumpy.core.generics import Sequential
from comnumpy.core.metrics import get_ser

# Build a 16-QAM communication chain
chain = Sequential([
    SymbolGenerator(M=16),
    SymbolMapper(M=16),
    AWGN(snr_dB=15),
    SymbolDemapper(M=16),
])

# Transmit 10,000 symbols and evaluate performance
tx_symbols, rx_symbols = chain(10000)
print(f"SER = {get_ser(tx_symbols, rx_symbols)}")
```

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/vincentchoqueuse/comnumpy.git
```

For development (editable mode):

```bash
git clone https://github.com/vincentchoqueuse/comnumpy.git
cd comnumpy
pip install -e .
```

## Features

| Module | Capabilities |
|--------|-------------|
| **core** | QAM/PSK mapping, AWGN channel, FIR filtering, pulse shaping, SER/BER metrics |
| **ofdm** | IFFT/FFT processing, cyclic prefix, carrier allocation, frequency-domain equalization, PAPR analysis |
| **mimo** | Rayleigh fading channel, ZF/MMSE/OSIC/ML detection, Monte Carlo evaluation |
| **optical** | Fiber propagation (SSFM), chromatic dispersion, Kerr nonlinearity, EDFA noise, digital back-propagation |

## Documentation

Full documentation with tutorials and API reference:

**[https://vincentchoqueuse.github.io/comnumpy/](https://vincentchoqueuse.github.io/comnumpy/)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up a development environment, coding standards, and how to add new submodules.

## License

This project is licensed under the [MIT License](LICENSE).
