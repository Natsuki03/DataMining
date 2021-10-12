import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
font_path = '/Library/Fonts/Arial Unicode.ttf'
font_prop = font_manager.FontProperties(fname = font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

path = "/Users/e195767/VS code/Data_Mining_Code/exam/"

#ex1.1

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
#plt.savefig(path + "ex1.1.png")
#plt.show()

#ex1.2
import random
import pandas as pd
random.seed(0)

n = [round(random.uniform(-1, 1), 2) for _ in range(20)]
true_val = [true_function(i) for i in n]

data = np.array([n, true_val]).T

df = pd.DataFrame(data, columns=["観測点", "真値"])

for i in range(len(n)):
    plt.plot(n[i], true_val[i], marker=".", markersize=10)
plt.savefig(path + "ex1.2.png")
plt.show()