import numpy as np
import numpy.matlib as matlib

def update(X, obs, R, state_dim, ens_size):

    """ This function is a simple EnKF analysis step implementation 

    It takes a state ensemble forcast vector X, m x N,  m ensemble size, N the state dimension, 
    an observation vector for the current analysis, obs size 1 x N, the observation noise 
    covariance matrix R, the state dimension and the ensemble size - here we take the 
    unperturbed observation operator to be the identity"""

    # generate forecast observations
    H = np.eye(state_dim)
    Y = H.dot(X.transpose()).transpose()

    # Anomaly matrix for X
    m_x = np.mean(X,axis=0)
    A_x = X - matlib.repmat(m_x,ens_size,1)

    # Anomaly matrix for Y
    m_y = np.mean(Y,axis=0)
    A_y = Y - matlib.repmat(m_y,ens_size,1)

    #Perturbing observation
    pert = np.random.multivariate_normal(np.zeros(state_dim),R,ens_size) #returns vector ens_size x state_dim
    pert_mean = np.mean(pert,0)
    
    #constraint perturbation to have zero mean
    pert = pert - matlib.repmat(pert_mean,ens_size,1)  
    obs_pert = matlib.repmat(obs,ens_size,1) + pert

    #EnKF Updating sample
    HCH_T = (A_y.transpose()).dot(A_y) # obs_dim x obs_dim
    CH_T = (A_x.transpose()).dot(A_y)  # state_dim x obs_dim
    K = CH_T.dot(np.linalg.inv(HCH_T + R)) #statistically, R should be replaced by D*D.transpose()
    Gain = K.dot(obs_pert.transpose()-Y.transpose())
    X = X + Gain.transpose()
        
    return(X)