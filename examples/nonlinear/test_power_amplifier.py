import matplotlib.pyplot as plt
import numpy as np
from comnumpy.core.devices import RappAmplifier, SalehAmplifier
from comnumpy.core.processors import Clipper
from comnumpy import style

style.use()

# reproduce Figure 4.9 of: Ghannouchi, Fadhel M., Oualid Hammi, and
# Mohamed Helaoui. "Behavioral modeling and predistortion of wideband
# wireless transmitters." John Wiley & Sons, 2015.

ampli_list = [
    Clipper(1),
    RappAmplifier(1, l=2),
    SalehAmplifier(1, alpha_am=2.1587, beta_am=1.1517, alpha_pm=4.033, beta_pm=9.1040)
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
