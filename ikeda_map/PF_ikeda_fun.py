#### PF-AUS Filter with Gassian resampling for the Lorenz 3 variable system ###
from pylab import *
from ikeda_map_v import IMV
from resamp_gauss import resamp_gauss

def PF(truth, obs, R,initial_cloud,u,tfin=2000,tanl=20):
 
    """Returns [mean_state, mean_error,avg_mean_error,rsn] for
    mandatory inputs truth trajectory, observations, and intitial cloud.
    
    [state_dimension, time_step]       = shape(truth)    
    [state_dimension, Nanl]            = shape(obs)
    [state_dimension, state_dimension] = shape(R)
    [state_dimension, particle_number] = shape(initial_cloud)
    scalar                             = type(u)
    [state_dimension, time_step]       = shape(mean_state)
    [state_dimension, time_step]       = shape(mean_error)
    [time_step]                        = shape(avg_mean_error)
    # of times resampler triggered     = rsn
    
    Optional arguments include the time step=h=0.01, sigma=a=10, beta=8/3, 
    rho=29, tanl=20.  The duration of the experiment tfin=2000, and 
    analysis time particle cloud is propagated between analysis times,
    at which point the observation is incorporated through the BAUS update
    and the Bayesian update of the weights.  Resampling is initiated
    according to resamp_gauss when Neff is less than the threshold."""
   
    # NUMBER OF ANALYSES
    Nanl = int(tfin/tanl)

    #number of resample steps
    rsn = 0
    #number of failsafe steps
    fsn = 0
    
    #Set initial trajectory matricies and weights
    [state_dimension, particle_number] = shape(initial_cloud)    
    part = zeros([state_dimension,tfin+1,particle_number])
    part[:,0,:] = initial_cloud
    part_weights = ones(particle_number)*(1.0/particle_number)
    Q = inv(R)
    mean_state = zeros([state_dimension,tfin+1])
    #corrections = zeros([3,Nanl])

    #loop over number of analyses
    for j in range(Nanl):

        #propagate each particle to next analysis step
        part[:,j*tanl:(j+1)*tanl+1,:] = IMV(tanl,part[:,j*tanl,:],u)
                                                     
        
        #Calculate Mean at each propagated state up to analysis time step
        weighted_cloud = part[:,j*tanl:(j+1)*tanl,:]*part_weights
        mean_state[:,j*tanl:(j+1)*tanl]=mean(weighted_cloud,2)*particle_number

        #Bayesian update is applied to particle wieghts    
        innov = obs[:,j] - part[:,(j+1)*tanl,:].T
        temp = array([sum(innov*Q[:,0],1),
                      sum(innov*Q[:,1],1)])
        part_weights = part_weights*exp(-0.5*sum(temp*innov.T,0))
        part_weights = part_weights/sum(part_weights)

        #Resampling step if number of effective samples falls below threshold                  
        n_eff = 1/(part_weights.dot(part_weights))
        if  (n_eff < 2) or isnan(n_eff):
            if isnan(n_eff):
                fsn = fsn + 1
                part_weights = ones(particle_number)*(1.0/particle_number)
            rsn = rsn + 1
            part[:,(j+1)*tanl,:] = resamp_gauss(part[:,(j+1)*tanl,:],
                                                    part_weights,.0004,
                                                    .01/particle_number)
        
                 
                                               
            part_weights = ones(particle_number)*(1.0/particle_number)
                                  
        #Find mean state updated weights      
        weighted_cloud = part[:,(j+1)*tanl,:]*part_weights             
        mean_state[:,(j+1)*tanl] = mean(weighted_cloud,1)*particle_number

    print(fsn)
    print(rsn - fsn)
    #mean_error_av = mean_error*mean_error
    mean_error = abs(mean_state - truth)
    error_dist = zeros(tfin+1)    
    avg_mean_error = zeros(tfin+1)
    for i in range(tfin+1):
        error_dist[i] = sqrt(mean_error[:,i].dot(mean_error[:,i]))
        avg_mean_error[i] = mean(error_dist[0:i+1])
    return(mean_state,mean_error, avg_mean_error)