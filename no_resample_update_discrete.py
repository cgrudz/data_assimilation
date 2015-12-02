import numpy as np
from pylab import find

def no_resample_update(weights,obs,Q,cloud,ens_size,state_dim):
    
    """This performs the analysis step of the particle filter without resampling.
    
    When a particle weight falls below the threshold, the particle is removed from
    the ensemble.  The analysis ensemble consists of the remaining particles with
    re-calculated weights, updated purely with the likelyhood function and the
    normalization of the weights in the new enesemble."""
    
    # vectorize the innovation from the integration form
    innov = -1*(cloud- obs).transpose()
        
    # compute the exponent of the likelyhood function
    temp  = np.sum(Q.dot(innov)*innov,axis=0)
    l_hood = np.exp(-0.5*temp)**(1.0/3.0)
        
    # update the weights
    weights = weights*l_hood
    weights = weights/(np.sum(weights))
        
    # delete the ensemble members falling below a threshold weight and re-weight
    deleting_idx = weights < 1e-20
    deleting_idx = find(deleting_idx)
    analysis = np.delete(cloud,deleting_idx,0)
    ens_size = len(analysis[:,0])
        
    weights = np.delete(weights,deleting_idx)
    weights = weights/(sum(weights))
        
    return [analysis,weights,ens_size]
