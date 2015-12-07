import numpy as np
from no_resample_update_discrete import no_resample_update

def  bootstrap(model,state_dim,prior,ens_size,thresh,steps,nanl,tanl,obs,Q):

    """This is the unbiased bootstrap particle filter function
    
    The fields to supply are the derivative function `model', the initial cloud
    'initial', the model dimension and cloud size,
    the particle weights at initialization `weights', the interval 
    to integrate the cloud '`interval', the observation with which to 
    update the cloud at the end analysis time, and the inverse of the 
    observational error covaraince `Q'.
    
    We assume that the model is vectorized to accept a cloud of initial
    conditions of the shape [model_dimension, cloud_size]."""
    
    # storage dictionary for the trajectories and weights
    p_series = {}
    A = 'A_'

    # divergence safety check
    divergence = False
    
    # define the initial weights
    weights = (1.0/ens_size)*np.ones(ens_size)    

    # loop through the analysis times starting at time zero
    for i in range(nanl):

        # store the prior weights and states
	prior_W = weights        
	prior_S = prior

        # recompute the weights, and throw out neglible particles
        [analysis,weights,ens_size] = no_resample_update(weights,thresh,obs[i,:],Q,prior,ens_size,state_dim)        
	post_S = analysis

        # check for filter divergence
        if ens_size < 10:
            divergence = True
            A_i = A + str(i)
            p_series[A_i] = {'prior':prior_S,'prior_weight':prior_W,'post':post_S,'post_weight':weights}
            break
        
        # map the cloud to the next analysis time;
        traj = model(analysis,steps,ens_size)
        
        #create storage for next iteration
        A_i = A + str(i)
        p_series[A_i] = {'prior':prior_S,'prior_weight':prior_W,'post':post_S,'post_weight':weights,'traj':traj}
        
        #initialize the next forecast
        prior = traj[:,-1,:]
        
    # final analysis time weight update - no forward trajectory to store
    if not divergence:
	prior_W = weights
	prior_S = prior
        [analysis,weights,ens_size] = no_resample_update(weights,thresh,obs[i+1,:],Q,prior,ens_size,state_dim)
	post_S = analysis        
	A_i = A + str(i+1)
        p_series[A_i] = {'prior':prior_S,'prior_weight':prior_W,'post':post_S,'post_weight':weights}
    
    return p_series
