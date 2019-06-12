from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# def f(x,y):
# 	return x*x+y*y

# def g(x,u):
# 	alpha = 100
# 	return x*x+u*u/alpha**2

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x_vec = np.linspace(-1.2,1.2,50)
# y_vec = np.linspace(-1,1,50)

# grid = [ [x,y] for x in x_vec for y in y_vec]
# z = np.zeros(2500)

# x = [p[0] for p in grid]
# y = [p[1] for p in grid]

# ax.plot(x, y, [g(x,y) for x in x_vec for y in y_vec], 'o')
# plt.show()


# from tqdm import tqdm
# for i in tqdm(range(10000000)):
# 	np.exp(i*i)