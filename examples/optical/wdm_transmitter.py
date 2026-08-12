"""A 5-channel WDM transmitter, from symbols to the full-field waveform.

The point of the example is the *shape*. One extra axis is created at
the very beginning -- ``(C, N_symbols)`` -- and every per-channel block
rides it: mapping, oversampling and pulse shaping all broadcast over the
channel axis and never learn that WDM exists. Multiplexing is then a
weighted sum over that axis, and demultiplexing is its inverse.

Run it to see the composite spectrum, the five slots, and the
adjacent-channel crosstalk the truncated pulse shape actually produces.
"""
import matplotlib.pyplot as plt
import numpy as np

from comnumpy.core import (SRRCFilter, Sequential, SymbolGenerator,
                           SymbolMapper, Upsampler)
from comnumpy.core.utils import Constellation
from comnumpy.optical import WDMDemultiplexer, WDMGrid, WDMMultiplexer

N_CHANNELS = 5
N_SYMBOLS = 512
OVERSAMPLING = 8
ROLL_OFF = 0.2
SYMBOL_RATE = 32e9                       # 32 Gbaud per channel
SPACING = 50e9                           # the ITU 50 GHz fixed grid

# The occupied bandwidth of an SRRC-shaped channel is (1 + rho) * Rs;
# the grid refuses to be built if that exceeds the spacing.
bandwidth = (1 + ROLL_OFF) * SYMBOL_RATE
fs = OVERSAMPLING * SYMBOL_RATE          # composite sampling rate

grid = WDMGrid.itu([-16, -8, 0, 8, 16], bandwidth_Hz=bandwidth,
                   center_Hz=193.1e12)
print(grid)
print(f"\ncomposite rate {fs / 1e9:.4g} GHz "
      f"(the grid needs at least {grid.min_fs / 1e9:.4g} GHz)\n")

# --- transmitter: the channel axis is created once, and never mentioned again
transmitter = Sequential([
    SymbolGenerator(16, seed=0, name="symbols"),
    SymbolMapper(Constellation("QAM", 16)),
    Upsampler(OVERSAMPLING),
    SRRCFilter(ROLL_OFF, OVERSAMPLING, N_h=10),
], name="per-channel transmitter")

x = transmitter((N_CHANNELS, N_SYMBOLS))       # (C, N)
y = WDMMultiplexer(grid, fs=fs)(x)             # (N,) -- the full field
print(f"per-channel {x.shape} -> composite {y.shape}")

# --- receiver: tune to the middle wavelength
recovered = WDMDemultiplexer(grid, fs=fs)(y)   # (C, N)

# How much of a neighbour lands in a channel's own band: light one
# channel only, and measure what comes out of the others.
lit = np.zeros_like(x)
lit[2] = x[2]
leak = WDMDemultiplexer(grid, fs=fs)(WDMMultiplexer(grid, fs=fs)(lit))
crosstalk_dB = 10 * np.log10(np.mean(np.abs(leak[1]) ** 2)
                             / np.mean(np.abs(leak[2]) ** 2))
print(f"adjacent-channel crosstalk: {crosstalk_dB:.1f} dB "
      f"(from the truncated pulse shape, not from the blocks)")

# --- figures
freq = np.fft.fftshift(np.fft.fftfreq(y.size, d=1 / fs)) / 1e9
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

spectrum = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(y))) + 1e-12)
ax_top.plot(freq, spectrum - spectrum.max(), lw=0.6)
ax_top.set(ylabel="dB", ylim=(-70, 5),
           title=f"{N_CHANNELS}-channel WDM composite field, "
                 f"{SYMBOL_RATE / 1e9:.0f} Gbaud on a "
                 f"{SPACING / 1e9:.0f} GHz grid")
for offset in grid.offsets_Hz:
    ax_top.axvspan((offset - bandwidth / 2) / 1e9,
                   (offset + bandwidth / 2) / 1e9, alpha=0.12)

for index in range(N_CHANNELS):
    channel = 20 * np.log10(
        np.abs(np.fft.fftshift(np.fft.fft(recovered[index]))) + 1e-12)
    ax_bottom.plot(freq, channel - spectrum.max(), lw=0.6,
                   label=f"channel {index}")
ax_bottom.set(xlabel="frequency offset from 193.1 THz (GHz)", ylabel="dB",
              ylim=(-70, 5), title="after demultiplexing (each at baseband)")
ax_bottom.legend(ncol=5, fontsize="small")
fig.tight_layout()
plt.show()
