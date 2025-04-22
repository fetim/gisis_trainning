import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import ricker

def ricker(t,fcut,tlag=0):
    """
    Ricker wavelet using cut off frequency.
    """
    pi = np.pi
    td = t - tlag
    fc = fcut / (3*np.sqrt(pi))
    return (1-2*pi*(pi*fc*td)**2)*np.exp(-pi*(pi*fc*td)**2)


Nx = 501
dx = 10
Nt = 1001
dt = 0.001

x = np.linspace(0,dx*(Nx-1),Nx)
t = np.linspace(0,dt*(Nt-1),Nt) # O tempo representará a profundidade nesse caso

t1,t2          = 0.3, 0.6 # tempos (profundidade) das interfaces
rho1,rho2,rho3 = 1000,2400,3400 # densidade de cada camada
vp1,vp2,vp3    = 1500,3000,4000 # velocidade de cada camada
z1,z2,z3       = rho1*vp1,rho2*vp2,rho3*vp3

# criando os perfis
rho = np.zeros(Nt)
vp  = np.zeros(Nt)

rho[:int(t1/dt)]           = rho1
rho[int(t1/dt):int(t2/dt)] = rho2
rho[int(t2/dt):]           = rho3

vp[:int(t1/dt)]           = vp1
vp[int(t1/dt):int(t2/dt)] = vp2
vp[int(t2/dt):]           = vp3

Z = vp * rho

plt.figure()
plt.subplot(1,3,1)
plt.plot(rho,t)
plt.xlabel('Densidade')
plt.gca().invert_yaxis()

plt.subplot(1,3,2)
plt.plot(vp,t)
plt.xlabel('Velocidade Compressional')
plt.gca().invert_yaxis()

plt.subplot(1,3,3)
plt.plot(Z,t)
plt.xlabel('Impedância acústica')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Criando uma secao 2D
z2D = np.zeros([Nt,Nx])
for ix in range(Nx):
    z2D[:,ix] = Z

# Calculando a impedância
plt.figure()
plt.imshow(z2D,cmap="jet",aspect="auto",extent=[x[0],x[-1],t[-1],t[0]])
plt.xlabel("distancia lateral (m)")
plt.ylabel("tempo (s)")
plt.title("Impedancia")
plt.colorbar()
plt.tight_layout()
plt.show()

# Calculando a refletividade
R = np.zeros(Nt)
for i in range(Nt-1):
    R[i] = (Z[i+1]-Z[i])/(Z[i+1]+Z[i])

plt.figure()
plt.plot(R,t)
plt.xlabel('Coeficiente de Reflexão')
plt.ylabel('Tempo')
plt.ylim(t[-1],t[0])
plt.show()

wavelet = ricker(t, 40,t[-1]/2) # wavelet de fase zero

# convolução com numpy
sinal = np.convolve(R,wavelet,mode='same')

# convolução com FFT
tshift = 0 #t[-1]/2
w = 2*np.pi*np.fft.fftfreq(Nt,dt)
R_fft = np.fft.fft(R)
wavelet_fft = np.fft.fft(wavelet)
sinal_fft =  R_fft * wavelet_fft * np.exp(-1.0j*w*tshift)
sinal_tempo = np.fft.fftshift(np.fft.ifft(sinal_fft))

plt.subplot(1,3,1)
plt.plot(wavelet,t)
plt.ylabel('Tempo')
plt.ylim(t[-1],t[0])

plt.subplot(1,3,2)
plt.plot(R,t)
plt.ylim(t[-1],t[0])

plt.subplot(1,3,3)
plt.plot(sinal,t,label='convolução numpy')
plt.plot(sinal_tempo,t,label='convolução FFT')
plt.ylim(t[-1],t[0])

plt.tight_layout()
plt.legend()
plt.show()
