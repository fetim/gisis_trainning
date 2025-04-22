import numpy as np
import matplotlib.pyplot as plt

def ricker(t,fcut,tlag=0):
    """
    Ricker wavelet using cut off frequency.
    """
    pi = np.pi
    td = t - tlag
    fc = fcut / (3*np.sqrt(pi))
    return (1-2*pi*(pi*fc*td)**2)*np.exp(-pi*(pi*fc*td)**2)


def ricker2(t,fc,tlag=0):
    """
    Ricker wavelet using central frequency.
    """
    pi = np.pi
    td = t - tlag
    return (1-2*(pi*fc*td)**2)*np.exp(-(pi*fc*td)**2)

Nt = 1001
dt = 0.001
t = np.linspace(0,Nt*dt,Nt,endpoint=False)

fcut = 60 # cut off frequency
wavelet = ricker(t,fcut,0.0)

fc = fcut /3 # central frequency is one thrid of cut off frequency
wavelet2 = ricker2(t,fc,0.0)


plt.figure()
plt.plot(t,wavelet,label=f'cut off frequency {fcut} Hz')
plt.plot(t,wavelet2,'--',label=f'central frequency {fc} Hz')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
