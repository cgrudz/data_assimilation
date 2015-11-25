from pylab import *

def IMS(steps,init_cond,u):
    """Returns a trajectory of particle cloud starting at init_cond for steps
    iterations of the Ikeda map, scalar version"""

    #trajectory vector defined
    tra = zeros([2,steps+1])
    
    tra[:,0] = init_cond 
    
    for i in range(steps):
        t_step = .4 - 6.0/(1 + sum(tra[:,i]*tra[:,i],0))
        tra[0,i+1] = 1.0+ u*(tra[0,i]*cos(t_step) - tra[1,i]*sin(t_step))
        tra[1,i+1] = u*(tra[0,i]*sin(t_step) + tra[1,i]*cos(t_step))

    return(tra)

def IMV(steps,init_cond,u):
    """Returns a trajectory of particle cloud starting at init_cond for steps
    iterations of the Ikeda map, vectorized version"""
    
    [state_dim, particle_number] = shape(init_cond)
    
    #trajectory vector defined
    tra = zeros([state_dim,steps+1,particle_number])
    
    tra[:,0,:] = init_cond 
    
    for i in range(steps):
        t_step = .4 - 6.0/(1 + sum(tra[:,i,:]*tra[:,i,:],0))
        tra[0,i+1,:] = 1.0+ u*(tra[0,i,:]*cos(t_step) - tra[1,i,:]*sin(t_step))
        tra[1,i+1,:] = u*(tra[0,i,:]*sin(t_step) + tra[1,i,:]*cos(t_step))
        
    return(tra)