import numpy as np
import skfuzzy as fuzz

class cluster():
  def __init__(self,x,y,U,n_clusters):
    data = np.reshape(U,(1,-1))
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data,n_clusters,2,error=0.0001, maxiter=10000, init=None)
    self.labels = np.reshape(np.argmax(u,axis=0),U.shape)
    self.labels = self.relabel(self.labels,cntr,n_clusters)
    self.ys = self.get_ys(self.labels,y,n_clusters)

  def relabel(self,label,center,n_clusters):
    tmp = np.linspace(0,n_clusters-1,n_clusters,dtype=np.int)
    center,tmp = zip(*sorted(zip(center,tmp)))
    xx,yy = np.shape(label)
    mask = np.zeros((xx,yy,n_clusters))
    for ii in range(n_clusters):
        mask[:,:,ii] = label == tmp[ii]
    for ii in range(n_clusters):
        label[np.nonzero(mask[:,:,ii])] = ii+1
    
    return label

  def get_ys(self,label,y,n_clusters):
    nx,ny = label.shape
    
    ys = np.zeros((nx,n_clusters-1))

    for n in range(n_clusters-1):
      for ii in range(nx):
        ytmp = np.array([])
        for jj in range(ny-1):
          if (label[ii,jj] == n+2 and label[ii,jj+1] == n+1) or (label[ii,jj] == n+1 and label[ii,jj+1] == n+2):
              ytmp = np.append(ytmp,0.5*(y[jj]+y[jj+1]))
        if len(ytmp) != 0:
            ys[ii,n] = np.max(ytmp)
        else:
            ys[ii,n] = 0

    return ys

