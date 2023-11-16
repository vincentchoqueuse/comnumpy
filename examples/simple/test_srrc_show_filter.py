import numpy as np
from dsp.frontend import SRRC_filter
import matplotlib.pyplot as plt


# check property of SRRC filter
# https://en.wikipedia.org/wiki/Root-raised-cosine_filter

N_h = 10
N = 2**15
oversampling = 50

legend_name = ["beta=1", "beta=0.5", "beta=0"]

x = (1/np.sqrt(2))*(np.random.randn(N)+1j*np.random.randn(N))
b = (0.5/np.sqrt(2))*(np.random.randn(N)+1j*np.random.randn(N))

for beta in [1, 0.5, 10**-10]:
    srrc = SRRC_filter(beta, oversampling, method="fft")
    
    y = srrc(x+b)
    print("Power In={}".format(np.mean(np.abs(x)**2)))
    print("Power Out={}".format(np.mean(np.abs(y)**2)))

    h = srrc.h()
    t = np.arange(len(h))/oversampling - N_h
    plt.figure(1)
    plt.plot(t, h)

    plt.figure(2)
    h2 = np.convolve(h,h)
    t = np.arange(len(h2))/oversampling - 2*N_h
    plt.plot(t, h2)

plt.figure(1)
plt.legend(legend_name)
plt.title("filter impulse response")

plt.figure(2)
plt.legend(legend_name)
plt.title("auto convolution")
plt.show()