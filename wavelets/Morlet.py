import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
t_min = -1
t_max = 1
num = int(t_max/dt)
t = np.linspace(t_min,t_max,num)
f = 30
psi = (1 - 2*(np.pi*f*t)**2)*np.exp(-(np.pi*f*t)**2)
plt.plot(t,psi)
plt.xlabel('Time, s')
plt.ylabel('Amplitude')
plt.show(block=False)

def dft(função):
    N = len(função)
    dft_result = []
    for k in range(N):
        real = sum(função[n] * np.cos(2 * np.pi * k * n / N) for n in range(N))
        imag = sum(-função[n] * np.sin(2 * np.pi * k * n / N) for n in range(N))
        amplitude = np.sqrt(real ** 2 + imag ** 2) / N
        dft_result.append(amplitude)
    return dft_result

# dft_result = dft(psi)

frequencias = []
for i in range (len(t)):
    frequencias.append(i/(t[-1] - t[0]))

frequencias_positivas = []
for i in range (len(frequencias)//2):
    frequencias_positivas.append(frequencias[i])

#%% Morlet Test
def morlet_wavelet(t,f0,width):
    gaussianWindow = np.exp(-(t**2)/(2*width**2))
    sineWave = np.cos(2*np.pi*f0*t)
    return gaussianWindow * sineWave

MorletReal = np.real(morlet_wavelet(t,30,0.05))

dft_result = dft(MorletReal)

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,MorletReal)
plt.title("Real part Wavelet Morlet")
plt.xlabel("time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(frequencias[:num//2],dft_result[:num//2])
plt.title("Amplitude Spectrum")
plt.xlabel("frequency (Hz)")
plt.ylabel("Amplitude")


plt.show(block=False)