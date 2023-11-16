import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from comnumpy.dsp.modem import get_alphabet

# plot data
sns.set_theme()

norm = True
modulation = "QAM"

for M in [4, 16, 32, 64]: 

    alphabet = get_alphabet(modulation, M, norm=norm)
    plt.figure()
    plt.plot(np.real(alphabet),np.imag(alphabet),"*")
    ax = plt.gca()
    M = int(np.log2(M))

    for index in range(len(alphabet)):
        txt = np.binary_repr(index,width=M)
        ax.annotate(txt, (np.real(alphabet[index]), np.imag(alphabet[index])))
    
    plt.axis("equal");
    plt.xlabel("Real Part")
    plt.ylabel("Imag Part")
    plt.title("Constellation {}-{}".format(modulation, M))

plt.show()