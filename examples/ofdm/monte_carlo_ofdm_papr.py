import numpy as np
import matplotlib.pyplot as plt
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.processors import Serial2Parallel
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ccdf
from comnumpy.ofdm.processors import CarrierAllocator, IFFTProcessor
from comnumpy.ofdm.metrics import compute_papr


img_dir = "../../docs/examples/img/"


N_sc = 1024
L = 10000
type, M = "PSK", 4
alphabet = get_alphabet(type, M)
alphabet_generator = np.arange(M)
papr_dB_threshold = np.arange(4, 13, 0.1)
gamma = 10**(papr_dB_threshold/10)
os = 4

carrier_type = np.zeros(os*N_sc)
carrier_type[:N_sc] = 1

chain = Sequential([
        SymbolGenerator(M),
        SymbolMapper(alphabet),
        Serial2Parallel(N_sc, name="s2p"),
        CarrierAllocator(carrier_type=carrier_type, name="carrier_allocator"),
        IFFTProcessor()
    ])

N = N_sc*os*L
y = chain(2**16)
y = np.ravel(y) # perform parallel2serial conversion (C-order flatten of (..., T, F) blocks)

papr_dB = compute_papr(y, unit="dB", axis=-1)
plt.plot(np.abs(y))
plt.ylabel("$|x[n]|^2$")
plt.xlabel("$n$ [sample]")
plt.title(f"PAPR={papr_dB:.2f}dB")
plt.savefig(f"{img_dir}/monte_carlo_ofdm_papr_fig1.png")

plt.figure()
N_sc_list = [256, 1024]

for N_sc in N_sc_list:

    carrier_type = np.zeros(os*N_sc)
    carrier_type[:N_sc] = 1

    chain["s2p"].set_N_sub(N_sc)
    chain["carrier_allocator"].set_carrier_type(carrier_type)

    N = N_sc*os*L
    y = chain(N)

    papr_dB_array = compute_papr(y, unit="dB", axis=-1)

    papr_dB, ccdf = compute_ccdf(papr_dB_array)
    plt.semilogy(papr_dB, ccdf, label=f"exp: N_sc={N_sc}")

    ccdf_theo = 1 - (1 - np.exp(-gamma))**(2.8*N_sc)
    plt.semilogy(papr_dB_threshold, ccdf_theo, label=f"theo: N_sc={N_sc}")


plt.ylim([1e-4, 1])
plt.xlim([6, 13])
plt.xlabel("PAPR (dB)")
plt.ylabel("CCDF")
plt.title("CCDFs of PAPR of an OFDM signal with 256 and 1024 subcarriers")
plt.grid()
plt.legend()

plt.savefig(f"{img_dir}/monte_carlo_ofdm_papr_fig2.png")
plt.show()
