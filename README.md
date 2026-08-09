# comnumpy

**A Python library for communication system prototyping and simulation.**

[![Tests](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/tests.yml/badge.svg)](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/tests.yml)
[![Docs](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/docs.yml/badge.svg)](https://github.com/vincentchoqueuse/comnumpy/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`comnumpy` provides Digital Signal Processing (DSP) algorithms for communication systems, covering MIMO, OFDM, and optical fiber simulations. It is designed for **researchers**, **engineers**, and **students** who want to simulate and analyze communication systems without implementing standard algorithms from scratch.

## Why comnumpy?

- **Modular design** — Build custom communication chains by combining reusable `Processor` blocks with `Sequential`, inspired by PyTorch’s `nn.Module` pattern.
- **Lightweight** — Around 400 KB of clean code. Only requires `numpy`, `scipy` and `matplotlib`.
- **Comprehensive** — Covers AWGN, OFDM, MIMO, and optical fiber channels with nonlinear propagation.
- **Well documented** — Tutorials with math, diagrams, and ready-to-run examples.

## Quick Example

```python
from comnumpy import (Sequential, SymbolGenerator, SymbolMapper,
                      SymbolDemapper, AWGN, compute_ser, get_alphabet)

# Build a 16-QAM communication chain: the module list describes the
# communication system, and `taps` names the signals to observe
alphabet = get_alphabet("QAM", 16)
chain = Sequential([
    SymbolGenerator(M=16, seed=42, name="tx"),
    SymbolMapper(alphabet),
    AWGN(snr_dB=15, seed=123),
    SymbolDemapper(alphabet),
], taps=["tx"])

# Transmit 10,000 symbols and evaluate performance
detected = chain(10_000)
print(f"SER = {compute_ser(chain.tap('tx'), detected)}")  # SER = 0.016
```

## Installation

```bash
pip install comnumpy
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

## Examples

Ready-to-run example scripts are available at:

**[https://github.com/vincentchoqueuse/comnumpy/tree/main/examples](https://github.com/vincentchoqueuse/comnumpy/tree/main/examples)**

## Documentation

Full documentation with tutorials and API reference:

**[https://vincentchoqueuse.github.io/comnumpy/](https://vincentchoqueuse.github.io/comnumpy/)**

Two normative documents govern the code itself:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the decision record. Code comments
  referring to "decision D25", "D40a"… point here.
- **[CONVENTIONS.md](CONVENTIONS.md)** — tensor layouts, axis categories, how to
  observe a signal inside a chain.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up a development environment, coding standards, and how to add new submodules.

## License

This project is licensed under the [MIT License](LICENSE).
