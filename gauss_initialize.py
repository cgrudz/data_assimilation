from pylab import *
from scipy.integrate import odeint

def gauss_initialize(model,state_dim,particle_number,spin,dt,p_cov,obs_var):
    """Create a Guassian distribution for initialization of the prior distribution for particle filter
    
    We arbitrarily initialize a vector of ones in the state space and spin this particle onto the 
    attractor.  After the particle is spun, we draw particles from a Guassian distribution with mean
    at a perturbation of the final state, and covariance given by p_cov.  This is returned along
    with the initial truth value for the dual experiment."""
    
    init = ones(state_dim)
    
    spun = odeint(model,init,spin)
    truth = spun[-1,:]
    
    mean = truth + multivariate_normal(zeros(state_dim),eye(state_dim)*obs_var)
    prior_cloud = multivariate_normal(truth,p_cov,[particle_number])
    
    return [truth,prior_cloud]
