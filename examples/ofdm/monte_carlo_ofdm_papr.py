import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator 
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ccdf
from comnumpy.ofdm.processors import CarrierAllocator, IFFTProcessor
from comnumpy.ofdm.metrics import compute_PAPR

# This script reproduces the first figure of the following article :
# - “An overview of peak-to-average power ratio reduction techniques for multicarrier transmission” by Han and Lee (2005).

img_dir = "../../docs/examples/img/"


# parameters
N_sc = 1024
L = 10000
type, M = "PSK", 4
alphabet = get_alphabet(type, M)
alphabet_generator = np.arange(M)
papr_dB_threshold = np.arange(4, 13, 0.1)
gamma = 10**(papr_dB_threshold/10)
os = 4

# defaut carrier type
carrier_type = np.zeros(os*N_sc)
carrier_type[:N_sc] = 1

# perform simulation
chain = Sequential([
        SymbolGenerator(M),
        SymbolMapper(alphabet),
        Serial2Parallel(N_sc, name="s2p"),
        CarrierAllocator(carrier_type=carrier_type, name="carrier_allocator"),
        IFFTProcessor()
    ])

# evaluate PAPR for one signal
N = N_sc*os*L
y = chain(2**16)
y = np.ravel(y, order="F") # perform parallel2serial conversion

papr_dB = compute_PAPR(y, unit="dB", axis=0)
plt.plot(np.abs(y))
plt.ylabel("$|x[n]|^2$")
plt.xlabel("$n$ [sample]")
plt.title(f"PAPR={papr_dB:.2f}dB")
plt.savefig(f"{img_dir}/monte_carlo_ofdm_papr_fig1.png")

plt.figure()
N_sc_list = [256, 1024]

for N_sc in tqdm(N_sc_list):
 
    # set S2P and carrier allocation
    carrier_type = np.zeros(os*N_sc)
    carrier_type[:N_sc] = 1

    chain["s2p"].set_N_sub(N_sc)
    chain["carrier_allocator"].set_carrier_type(carrier_type)

    # generate data
    N = N_sc*os*L
    y = chain(N)

    # evaluate metric
    papr_dB_array = compute_PAPR(y, unit="dB", axis=0)

    # display experimental curves
    papr_dB, ccdf = compute_ccdf(papr_dB_array)
    plt.semilogy(papr_dB, ccdf, label=f"exp: N_sc={N_sc}")

    # display theoretical curves
    ccdf_theo = 1 - (1 - np.exp(-gamma))**(N_sc*os)
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
