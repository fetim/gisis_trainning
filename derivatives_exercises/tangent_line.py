"""
viewing the tangent line in a function
"""
import GISIS_library
import numpy as np
import matplotlib.pyplot as plt

#start class
function = GISIS_library.polynomns()

# create the function
function.select_function("sum_2_sines",a=3,b=4,c=1,d=20)

plt.figure()
# movie to view the tangent line in all points
for i in range(0,function.N,function.step):
    function.viewGraph()
    # find the tangent line
    function.tangent_line_evalution(i)
    function.viewTangentLine()
    