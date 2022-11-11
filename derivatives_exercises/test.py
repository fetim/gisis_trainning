import GISIS_library
import numpy as np
import matplotlib.pyplot as plt


test1 = GISIS_library.polynomns()
# test1.polynomn1(2,1)
# test1.polynomn2(1,0,0)
# test1.polynomn3(1,0,0,0)
test1.sine(30)
plt.figure()
for i in range(0,1000,10):
    test1.viewGraph()
    plt.ylim(-2,2)
    test1.tangent_line(i)
    plt.plot(test1.x,test1.df_dx)
    plt.pause(0.2)
    plt.show(block=False)
    plt.clf()