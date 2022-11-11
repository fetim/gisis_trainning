import numpy as np
import matplotlib.pyplot as plt

class polynomns:
    def __init__(self):
        self.x = np.linspace(-1,1,1000)
        self.y = np.zeros(len(self.x))
        pass
    
    def viewGraph(self):
        plt.plot(self.x,self.y)
        plt.grid()
        plt.show(block=False)

    def polynomn1(self,a,b):
        self.y = a*self.x + b
        pass

    def polynomn2(self,a,b,c):
        self.y = a*self.x*self.x + b*self.x + c
        pass

    def polynomn3(self,a,b,c,d):
        self.y = a*self.x*self.x*self.x + b*self.x*self.x + c*self.x + d
        pass

    def sine(self,freq):
        self.y =  np.sin(freq/2/np.pi * self.x)

    def tangent_line(self,idx):
        a = (self.y[idx+1] - self.y[idx])/(self.x[idx+1]-self.x[idx]) 
        self.df_dx = a * (self.x - self.x[idx])  + self.y[idx]
        pass

    

        