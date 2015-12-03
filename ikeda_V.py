import numpy as np

def ikeda(state,steps,particle_number):
    
    """This is the vectorized iterative map for the Ikeda model"""
    # Define the system parameters
    u = .83    

    # create storage for the trajectories
    traj = np.zeros([particle_number,steps+1,2])

    # unpack the state variables
    if particle_number == 1:
        X = state[0]
        Y = state[1]
    else:
        X = state[:,0]
        Y = state[:,1]
    traj[:,0,0] = X
    traj[:,0,1] = Y

    for i in range(steps):
        T = 0.4 - 6.0*(1 + X**2 + Y**2)**(-1)
        X1 = 1 + u*(X*np.cos(T) - Y*np.sin(T))
        Y1 = u*(X*np.sin(T) + Y*np.cos(T))
        traj[:,i+1,0] = X1
        traj[:,i+1,1] = Y1
	X = X1
	Y = Y1
    
    return traj
