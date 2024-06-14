import numpy as np
import matplotlib.pyplot as plt

A1 = 1
A2 = 1
A3 = 1
f1 = 20 #hz
f2 = 50 #hz
f3 = 100 #hz
omega1 = 2*np.pi*f1
omega2 = 2*np.pi*f2
omega3 = 2*np.pi*f3
dt = 0.004 # sample rate
T = 5 # period
nt = int(T/dt) + 1
time = np.linspace(0,T,nt)


sen1  = A1 * np.sin(omega1 * time)
sen2  = A2 * np.sin(omega2 * time)
sen3  = A3 * np.sin(omega3 * time)
sen_total = sen1 + sen2 + sen3


plt.figure()
plt.subplot(2,2,1)
plt.plot(time,sen1)
plt.subplot(2,2,2)
plt.plot(time,sen_total)

# Fourier Transform
frequency = np.fft.fftfreq(nt,dt)
Fsen1 = np.fft.fft(sen1)
Fsen2 = np.fft.fft(sen2)
Fsen3 = np.fft.fft(sen3)
Fsen_total = np.fft.fft(sen_total)

# amplitude_spectrum = np.real(np.conj(Fsen) * Fsen)
plt.subplot(2,2,3)
plt.plot(frequency,np.real(Fsen1 * np.conj(Fsen1)))
plt.subplot(2,2,4)
plt.plot(frequency,np.real(Fsen_total * np.conj(Fsen_total)))
plt.tight_layout()
plt.show(block=False)
