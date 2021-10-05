import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
font_path = '/Library/Fonts/Arial Unicode.ttf'
font_prop = font_manager.FontProperties(fname = font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

def true_function(x):
    '''
    >>> true_function(0)
    0.0
    '''
    y = np.sin(np.pi * x * 0.8) * 10
    return y

"""
#doctest
import doctest
doctest.testmod()
"""
x = np.arange(-1,1,0.01)

plt.plot(x,true_function(x),label="y = $\sin(\pi * x * 0.8) * 10$")
plt.title("ture_function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("ex1.1.png")
plt.show()