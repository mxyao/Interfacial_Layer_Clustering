import numpy as np
import matplotlib.pyplot as plt
import h5py

from kde import kde

f = h5py.File('channel5200_0-0.h5','r')

temp = f['u00000'][...]
U_temp = temp[0,:,:,0].T

x = np.linspace(0,8*np.pi,10240)

y = np.loadtxt('channel5200_yloc.dat')
ny = len(y) // 2
y = y[:ny]

nx1 = 0
nx2 = 200
x = x[nx1:nx2]
U = U_temp[nx1:nx2,:]

kde(U)