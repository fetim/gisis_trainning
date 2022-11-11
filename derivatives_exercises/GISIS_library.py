import numpy as np
import matplotlib.pyplot as plt

class polynomns:
    def __init__(self,x0=-1,xf=1,N=1000):
        self.x0   = x0
        self.xf   = xf
        self.N    = N
        self.step = int(N*0.01)
        self.x    = np.linspace(self.x0,self.xf,self.N)
        self.y    = np.zeros(self.N)
        pass
    
    def viewGraph(self):
        plt.plot(self.x,self.y)
        plt.ylim(self.y_min,self.y_max)
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

    def sine(self,Nlambda=1):
        self.y =  np.sin(Nlambda * np.pi * self.x)
        pass

    def sum_2_sines(self,Nlambda1=1,Nlambda2=2):
        self.y =  np.sin(Nlambda1 * np.pi * self.x) + np.sin(Nlambda2 * np.pi * self.x)

    def limits_function(self):        
        
        amplitude = np.abs(np.max(self.y) - np.min(self.y))
        self.y_min = np.min(self.y) - 0.10*amplitude
        self.y_max = np.max(self.y) + 0.10*amplitude
        pass

    def select_function(self,function_type="sine",a=1,b=0,c=0,d=0):
        """
        Select function to calculate the tangent straight line.\n
        function_type = sine (default), sum_2_sines, polynomn2, polynomn3 \n
        a = 1st coeficient\n
        b = 2nd coeficient\n
        c = 3rd coeficient\n
        d = 4th coeficient\n
        """
        if function_type == "sine":
            self.sine(a)
        
        elif function_type == "sum_2_sines":
            self.sum_2_sines(a,b)
            
        elif function_type == "polynomn2":
            self.polynomn2(a,b,c)
        
        elif function_type == "polynomn3":
            self.polynomn3(a,b,c,d)

        else:
            raise ValueError("Select function problem!\nfunction_type = %s is not defined"%function_type)
        
        self.limits_function()

    def tangent_line_evalution(self,idx):
        a = (self.y[idx+1] - self.y[idx])/(self.x[idx+1]-self.x[idx]) 
        self.tangent_line = a * (self.x - self.x[idx])  + self.y[idx]
        pass

    def viewTangentLine(self):
        plt.plot(self.x,self.tangent_line)
        plt.pause(0.05) # viewing in slow motion
        plt.show(block=False)
        plt.clf() # clean the plot
    

        