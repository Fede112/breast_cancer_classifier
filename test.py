from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


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


class A():
	exposure = 4
	def __init__(self):
		self.toto = 3

	@staticmethod
	def foo():
		print(f"aqui me pongo a cantar")



edu = A()

print(edu.toto)
edu.foo()





# print(a.reshape(2,-1).mean(-1))


d = {('a','b'):[1,2,3,4], ('a','c'):[5,6,7,8], ('z','b'):[9,10,11,12], ('z','c'):[13,14,15,16]}
df = pd.DataFrame(d, index=['r1','r2','r3','r4'])
print (df)
print(df.columns)
df.columns.names = ('l1','l2')
print(df.columns.values)
print (df)
# print (df.T.reset_index())
# print (df.T.reset_index().groupby('l1').mean())

# print (df['a'].groupby('b').mean())


# norm = {'a':[1,2,3,4], 'b':[5,6,7,8], 'c':[9,10,11,12], 'd':[13,14,15,16]}
# dnorm = pd.DataFrame(norm)
# dnorm.columns.names = ['l1']
# print(dnorm.columns)