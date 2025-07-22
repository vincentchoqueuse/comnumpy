import matplotlib.pyplot as plt
import numpy as np
from comnumpy.nonlinear.models import RappAmplifier, Clipping_Amplifier, SalehAmplifier

# reproduce the figure Figure 4.9 of the book Ghannouchi, Fadhel M., Oualid Hammi, and Mohamed Helaoui. Behavioral modeling and predistortion of wideband wireless transmitters. John Wiley & Sons, 2015.
    
ampli_list = [
    Clipping_Amplifier(1),
    RappAmplifier(1, p=2),
    SalehAmplifier(1,  2.1587, 1.1517, 4.033, 9.1040)
    ]

x = np.arange(0, 2, 0.01)

for ampli in ampli_list:
    y = ampli(x)
    plt.plot(x, np.abs(y), label=f"ampli={ampli.name}")

plt.legend()
plt.xlabel("Input Amplitude")
plt.ylabel("Output Amplitude")
plt.grid()
plt.show()



