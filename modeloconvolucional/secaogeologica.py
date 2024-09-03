import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker

x = np.linspace(0, 6, 500)
y = np.linspace(0,10,1331)

camada1,camada2,camada3 = np.full_like(x, 3),np.full_like(x, 6), np.full_like(x, 10)

plt.fill_between(x, 0, camada1, color='orange', label='Camada 1')
plt.fill_between(x, camada1, camada2, color='yellow', label='Camada 2')
plt.fill_between(x, camada2,camada3, color='brown', label='Camada 3')
plt.xlabel('Distância horizontal')
plt.ylabel('Profundidade')
plt.gca().invert_yaxis()
plt.title('Seção Geológica')
plt.legend()
plt.show()

p_1,p_2,p_3= 2200,2400,2700
v1,v2,v3 = 2500,3000,3500
z1,z2,z3 = p_1*v1,p_2*v2,p_3*v3
Z = y.copy()
Z[(Z>=0)&(Z<=3)] = z1
Z[(Z>=3)&(Z<=6)] = z2
Z[(Z>=6)&(Z<=10)] = z3

plt.plot(Z,y)
plt.xlabel('Impedância acústica')
plt.ylabel('Profundidade')
plt.title('Perfil de Impedância Acústica das Camadas')
plt.gca().invert_yaxis()
plt.show()

r12 = (z2 - z1) / (z2 + z1)
r23 = (z3 - z2) / (z3 + z2)
R = np.zeros_like(y)
R[np.isclose(y, 3, atol=0.01)] = r12
R[np.isclose(y, 6, atol=0.01)] = r23

plt.plot(R,y)
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