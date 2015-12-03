import numpy as np
from scipy.integrate import odeint

def gen_obs(model,truth,H,exp_len,nanl,tanl,obs_var):
    
    """This function generates a sequence of observations for analysis
    
    Given the model, the initialization of the truth, observational operator H,
    the interval of the experiment integration, the number of analyses, the 
    time interval between analyses, and the vector of observational variances (assuming
    covariances are zero), this function returns a sequence of observations given by perturbations
    of the the true trajectory."""
    
    # Define the model and obs dimensions
    [obs_dim,state_dim] = np.shape(H)
    
    # Propagate a `true' trajectory to generate the observations
    truth = model(truth,exp_len,1)
    truth = truth.squeeze()

    # Create observations with error, with first observation at time zero, and ranging to
    # time exp_len, at intervals of tanl
    Q = np.linalg.inv(np.eye(state_dim)*np.sqrt(obs_var))
    obs = (H.dot(truth[::tanl,:].transpose())).transpose() + (np.random.randn(nanl+1,state_dim)*np.sqrt(obs_var))
    
    return [truth,obs,Q]
