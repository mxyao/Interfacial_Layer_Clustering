import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.stats import gaussian_kde

class kde():
    def __init__(self,U):
        data = np.reshape(U,(1,-1))

        xx = np.linspace(np.min(data),np.max(data),500)
        kernel = gaussian_kde(data,bw_method='scott')
        bins = 25
        plt.hist(data.T,bins,normed=True,edgecolor='black',facecolor='white')
        plt.plot(xx,kernel.evaluate(xx),color='k')
        plt.show()

