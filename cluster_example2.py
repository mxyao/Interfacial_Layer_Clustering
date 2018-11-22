import numpy as np
import matplotlib.pyplot as plt
import h5py

from cluster import cluster

f = h5py.File('channel5200_0-0.h5','r')

temp = f['u00000'][...]
U_temp = temp[0,:,:,0].T
V_temp = temp[0,:,:,1].T

x = np.linspace(0,8*np.pi,10240)

y = np.loadtxt('channel5200_yloc.dat')
ny = len(y) // 2
y = y[:ny]

nx1 = 0
nx2 = 200
x = x[nx1:nx2]
U = U_temp[nx1:nx2,:]
V = V_temp[nx1:nx2,:]

X,Y = np.meshgrid(x,y,indexing='ij')

n_clusters = 4

trial2 = cluster(x,y,U,n_clusters)

plt.contourf(X,Y,U,250,cmap='RdBu')
plt.contour(x,y,trial2.labels.T,3,colors='k')
plt.plot(x,trial2.ys[:,0],color='red',linestyle='--')
plt.plot(x,trial2.ys[:,1],color='red',linestyle=':')
plt.plot(x,trial2.ys[:,2],color='red',linestyle='-.')
plt.show()