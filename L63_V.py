import numpy as np

def L63(state,t):
    
    """This is the vectorized derivative for the Lorenz 63 model"""
    # Define the system parameters
    sigma = 10.0
    rho   = 28.0
    beta  = 8.0/3.0
    
    # Reshape the state vector to apply the derivative  
    particles = len(state)/3
    state = np.reshape(state,[particles,3])
    
    # unpack the state variables
    X = state[:,0]
    Y = state[:,1]
    Z = state[:,2]

    dx = sigma*(Y-X)
    dy = X*(rho - Z) - Y
    dz = X*Y - beta*Z
    
    deriv = np.array([dx,dy,dz]).transpose()
    deriv = np.reshape(deriv,particles*3)
    
    return deriv
