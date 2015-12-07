import numpy as np
from resample_systematic import resample_systematic as RS
from pylab import find

def SIR_RS(weights,obs,Q,cloud,ens_size,state_dim, Nmin = 1/float(ens_size)):
    
    """This performs the analysis step of the particle filter with SIR algorithm."""
    
    # vectorize the innovation from the integration form
    innov = -1*(cloud- obs).transpose()
    
    # compute the exponent of the likelyhood function
    temp  = np.sum(Q.dot(innov)*innov,axis=0)
    l_hood = np.exp(-0.5*temp)**(1.0/3.0)
    
    # update the weights and resample
    weights = weights*l_hood
    Neff = sum(weights*weights)
    
    # check for weights degeneracy as safety
    if Neff > 0:
        # if nondegenerate see if the effective number of particles is too low
        if 1/Neff < Nmin:
            resampled = RS(weights)
            weights = ones(ens_size)/float(ens_size)
    else:
        # if completely degenerate resample uniformly from the distribution
        weights = ones(ens_size)/float(ens_size)
        resampled = RS(weights)
            
    
    weights = np.ones(ens_size)/float(ens_size)
        
    return [analysis,weights,ens_size]
