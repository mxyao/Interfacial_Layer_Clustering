import numpy as np
import matplotlib.pyplot as plt

from cluster import cluster

filename = 'snapshot1.dat'
X,Y,U,V = np.loadtxt(filename,delimiter=',',unpack=True)
X = np.reshape(X,(86,216)).T
Y = np.reshape(Y,(86,216)).T
U = np.reshape(U,(86,216)).T
V = np.reshape(V,(86,216)).T

x = X[:,0]
y = Y[0,:]

n_clusters = 3

trial1 = cluster(x,y,U,n_clusters)

plt.contourf(X,Y,U,250,cmap='RdBu')
plt.contour(x,y,trial1.labels.T,2,colors='k')
plt.plot(x,trial1.ys[:,0],color='red',linestyle='--')
plt.plot(x,trial1.ys[:,1],color='red',linestyle=':')
plt.show()