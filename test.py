#!/usr/bin/python3


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import multiprocessing
def f(x):
	print(multiprocessing.current_process())
	print(x)
	# exit()
	return x*x


# with multiprocessing.Pool(1) as p:
# 	print (p.map(f, range(6)))


# p = multiprocessing.Pool(1)
# print (p.map(f, range(6)))

# a = input()
# b = input()

# assert a == b, "they are not equal" 

def f(x,y):
	return x*x + y*y


def g(x,u):
	alpha = 1000
	return x*x + u*u/alpha



x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
u = np.linspace(0,1,100)


fig = plt.figure()

ax = fig.add_subplot(111, projection = '3d')

ax.plot(x,y,1)

plt.show()