import numpy as np
from resample_systematic import resample_systematic as RS
from pylab import find

def SIR_update(weights,Nmin,tuning,resample_count,obs,Q,cloud,ens_size,state_dim):
    
    """This performs the analysis step of the particle filter with SIR algorithm."""
    
    # vectorize the innovation from the integration form
    innov = -1*(cloud- obs).transpose()
    
    # compute the exponent of the likelyhood function
    temp  = np.sum(Q.dot(innov)*innov,axis=0)
    l_hood = np.exp(-0.5*temp)**(1.0/3.0)
    
    # update the weights and resample
    weights = weights*l_hood
    weights = weights/np.sum(weights)
    Neff = sum(weights*weights)
    
    # check for weights degeneracy as safety
    if Neff > 0:
        # if nondegenerate see if the effective number of particles is too low
        if 1/Neff < Nmin:
            cov = np.cov(cloud.transpose())
            resampled = RS(weights)
            weights = np.ones(ens_size)/float(ens_size)
            cloud = cloud[resampled.astype(int),:] + np.random.multivariate_normal([0,0],tuning*cov,ens_size)
            resample_count = resample_count +1
    else:
        # if completely degenerate resample uniformly from the distribution
        cov = np.cov(cloud.transpose())
        weights = np.ones(ens_size)/float(ens_size)
        resampled = RS(weights)
        cloud = cloud[resampled.astype(int),:] + np.random.multivariate_normal([0,0],tuning*cov,ens_size)
        resample_count = resample_count +1            
        
    return [cloud,weights,resample_count]
