import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from comnumpy import style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.processors import Upsampler, DelayRemover
from comnumpy.core.utils import Constellation, hard_projector
from comnumpy.mimo.channels import SelectiveMIMOChannel, AWGN
from comnumpy.mimo.utils import rayleigh_channel
from comnumpy.mimo.compensators import BlindDualMIMOCompensator

style.use()


@dataclass
class CustomBlindDualMIMOCompensator(BlindDualMIMOCompensator):

    commuting_steps: tuple =  (20000, 500000)
    debug: bool = False

    def process_after_iteration(self, n_step, Y_sub):
        if (self.mode == "cma") and (n_step >= self.commuting_steps[0]):
            self.mode = "rde"
        if (self.mode == "rde") and (n_step >= self.commuting_steps[1]):
            self.mode = "dd"
            self.process_phase_correction(Y_sub)

    def process_phase_correction(self, Y_sub):
        phase_search_grid = np.arange(-np.pi/4, np.pi/4, 0.01)
        rotation_factor = np.exp(1j*phase_search_grid).reshape(-1, 1)

        for index in range(2):
            y_sub = Y_sub[index, :]
            _, data_est = hard_projector(rotation_factor * y_sub, self.alphabet)
            error = np.mean(np.abs(data_est - rotation_factor*y_sub)**2, axis=1)
            index_min = np.argmin(error)
            self.H_[index, :] *= np.conjugate(rotation_factor[index_min]) # be carefull, we apply the conjugate of H for compensation

            if self.debug:
                y_sub_corrected = rotation_factor[index_min] * y_sub
                plt.figure()
                plt.plot(error)

                plt.figure()
                plt.plot(np.real(y_sub), np.imag(y_sub), ".")
                plt.plot(np.real(y_sub_corrected), np.imag(y_sub_corrected), ".")
                plt.show()


# parameters
constellation = Constellation("QAM", 16)
N_t, N_r, N_tap = 2, 2, 3
N = 100000
N_h = 32
oversampling = 2
rolloff = 0.2
sigma2n = 1e-3
commuting_steps = (20000, 90000)

# generate channel
H_list = []
size = (N_t, N_t)
rng = np.random.default_rng()
H_array = rayleigh_channel(N_r, N_t, L=N_tap, rng=rng)

# create chain
chain = Sequential([
            SymbolGenerator(constellation.order),
            SymbolMapper(constellation),
            Upsampler(oversampling),
            SRRCFilter(rolloff, oversampling, N_h=N_h),
            SelectiveMIMOChannel(H=H_array, name="channel"),
            AWGN(sigma2=sigma2n, name="noise"),
            SRRCFilter(rolloff, oversampling, N_h=N_h),
            DelayRemover(delay=N_h*4),
            CustomBlindDualMIMOCompensator(L=9, alphabet=constellation, mu=1e-4, oversampling=oversampling,
                                           commuting_steps=commuting_steps,
                                           # the block handed to process_after_iteration:
                                           # the phase search below needs enough symbols
                                           # to decide on, not the default 20
                                           sub_block_length=500, name="filter"),
            DelayRemover(delay=int(0.9*N), name="tail"),
            ], taps=["filter", "tail"])

# simulate communication
Y = chain((N_t, N))

data = chain.tap("filter")  # full CMA output (before tail removal)
symbols = chain.tap("tail")
_, ax = plt.subplots()
ax.plot(np.real(symbols), np.imag(symbols), ".")
ax.set_title("after CMA convergence")
style.apply(ax, "iq")

# compute CMA loss
kernel_size = 100
R = (np.mean(np.abs(constellation.alphabet) ** 4)
     / np.mean(np.abs(constellation.alphabet) ** 2))
kernel = np.ones(kernel_size) / kernel_size
loss_cma = np.mean((np.abs(data)**2 - R)**2, axis=0)
loss_cma = np.convolve(loss_cma, kernel, mode='valid')

# compute RDE loss
radius_list = np.unique(np.abs(constellation.alphabet))
_, radius_est = hard_projector(np.abs(data), radius_list)
loss_rde = np.mean((radius_est**2 - np.abs(data)**2)**2, axis=0)
loss_rde = np.convolve(loss_rde, kernel, mode='valid')

# compute DD loss
_, data_est = hard_projector(data, constellation)
loss_dd = np.mean(np.abs(data_est - data)**2, axis=0)
loss_dd = np.convolve(loss_dd, kernel, mode='valid')

plt.figure()
plt.plot(loss_cma, label="CMA-loss")
plt.plot(loss_rde, label="RDE-loss")
plt.plot(loss_dd, label="DD-loss")
plt.axvline(x=commuting_steps[0], color="k")
plt.axvline(x=commuting_steps[1], color="k")
plt.ylabel("loss function")
plt.ylim([0, 1])
plt.legend()
plt.show()
