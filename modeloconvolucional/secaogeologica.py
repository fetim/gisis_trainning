import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker


Nx = 501
dx = 10
Nt = 1001
dt = 0.001

x = np.linspace(0,dx*(Nx-1),Nx)
t = np.linspace(0,dt*(Nt-1),Nt) # O tempo representará a profundidade nesse caso

t1,t2          = 0.3, 0.6 # tempos (profundidade) das interfaces
rho1,rho2,rho3 = 2200,2400,2700 # densidade de cada camada
vp1,vp2,vp3    = 2500,3000,3500 # velocidade de cada camada
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


# Criando uma secao 2D
z2D = np.zeros([Nt,Nx])
for ix in range(Nx):
    z2D[:,ix] = Z

plt.figure()
plt.imshow(z2D,cmap="jet",aspect="auto",extent=[t[-1],t[0],x[0],x[-1]])
plt.xlabel("distancia lateral (m)")
plt.ylabel("tempo (s)")
plt.title("Impedancia")
plt.colorbar()
plt.tight_layout()
plt.show()

##### REFAZER daqui para baixo
r12 = (z2 - z1) / (z2 + z1)
r23 = (z3 - z2) / (z3 + z2)
R = np.zeros_like(t)
R[np.isclose(t, 3, atol=0.01)] = r12
R[np.isclose(t, 6, atol=0.01)] = r23

plt.plot(R,t)
plt.xlabel('Coeficiente de Reflexão')
plt.ylabel('Profundidade')
plt.xlim(-0.2,0.2)
plt.gca().invert_yaxis()
plt.show()

t1 = 2*3/v1
t2 = 2*6/v2
t3= 2*10/v3
print(t1,t2,t3)
t = np.linspace(0,0.005,500)
R_tempo = np.zeros_like(t)
R_tempo[np.isclose(t, t1, atol=0.0001)] = r12
R_tempo[np.isclose(t, t2, atol=0.0001)] = r23

plt.plot(R_tempo,t)
plt.xlabel('Coeficiente de reflexão')
plt.ylabel('Tempo')
plt.gca().invert_yaxis()
plt.show()

wavelet = ricker(500, 4)
sinal = np.convolve(R_tempo,wavelet,mode='same')

plt.plot(t, sinal)
plt.ylabel('Sinal Sísmico')
plt.xlabel('Tempo')
plt.show()

R_tempo_fft = np.fft.rfft(R_tempo)
wavelet_fft = np.fft.rfft(wavelet)
sinal_fft = R_tempo_fft * wavelet_fft
sinal_tempo = np.fft.irfft(sinal_fft)

plt.plot(t,sinal_tempo)
plt.ylabel('Sinal Sísmico')
plt.xlabel('Tempo')
plt.show()