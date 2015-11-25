def L63_vectorized(state,t):
    
    """This is the vectorized derivative for the Lorenz 63 model"""

    # Define the system parameters
    sigma = 10
    rho   = 8.0/3.0
    beta  = 28
    
    # Reshape the state vector to apply the derivative  
    particles = len(state)/3
    state = reshape(state,[particles,3])
    
    # unpack the state variables
    X = state[:,0]
    Y = state[:,1]
    Z = state[:,2]

    dx = sigma*(Y-X)
    dy = X*(rho - Z) - Y
    dz = X*Y - beta*Z
    
    deriv = array([dx,dy,dz]).transpose()
    deriv = reshape(deriv,particles*3)
    
    return deriv
    
    
       
       
