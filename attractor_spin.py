from pylab import *
from scipy.integrate import odeint

def attractor_spin(particle_number,root,model,state_dim,obs_var):
    
    """Spin up the ensemble of initial conditions for the filter and `truth'
    
    Assign cubes of initial conditions for the ensemble spin up, run the spin,calculate 
    spin mean, and variance to set initial conditions for the experiment.  The number
    of particles must be a cubic, and the cube root of the number of particles is supplied
    as `root'.""" 

    # Generate basepoint for the cube of initial conditions
    ens_base = randint(-100,100)

    # Define cube of initial conditions for the spin up 
    cube = zeros([particle_number,state_dim])

    for i in range(root):
        for j in range(root):
            for k in range(root):            
                cube[(root**2)*i + root*j + k, :] =(array([k*exp(-1),j*exp(-1),i*exp(-1)])+ens_base)
                                                      

    # Spin up the initial cloud
    spin_cloud = reshape(cube,3*particle_number)
    spin_cloud = odeint(L63,spin_cloud,spin)
    spin_cloud = reshape(spin_cloud,[len(spin),particle_number,state_dim])
    spun_cloud = spin_cloud[-1,:,:]

    ## Calcualte the environmental statistics

    #Determine the mean for the spin cloud at each time step
    spin_mean = mean(spun_cloud,axis=0)

    #Calculate variance of the mean along spin up
    cl_var = 2*var(spun_cloud,axis=0)
    obs_var = obs_var*cl_var

    #Observational Error covariance stored    
    R = eye(state_dim)*obs_var
    Q = inv(R)

    ## Create the truth for `perfect' model experiment

    # Generate random ensemble member to be initializaiton for `truth'
    P = randint(0,particle_number)
    truth = spun_cloud[P,:]

    # `Truth' state is eliminated from the particle cloud
    PF_cloud = delete(spun_cloud,P,0)
    
    return [truth,PF_cloud,Q]
