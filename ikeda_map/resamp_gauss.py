from pylab import *    

def resamp_gauss(parts,weights,cov_fac,weight_min):
    
    """A generic resample algorithm, using Gaussian clouds about particles
    
    This resampler will reproduce samples not at the exact location of existing
    samples but in a Gaussian cloud of covariance which is a fraction cov_fac
    of the original covariance (useful for deterministic models, 
    or for EnKF etc.)

    cov_fac = 0.0004 ===> 2% std dev
    cov_fac = 0.0025 ===> 5% std dev

    [state_dimension, particle_number] = shape(parts)
    [particle_number]                  = shape(weights)

    weights less than weight_min are set equal to zero, particle is eliminated
    (useful in PF update step -- use weight_min ~ 0.1/particle_number)"""
    
    #make all weights falling below the threshold equal to zero
    [s_dim, particle_number] = shape(parts)
    kill_index = weights < weight_min
    #Rational: weights are scaled so their sum is equal to 1, so we will
    #scale the weihts by order particle_number**2 so that the reduced number
    #of weights times particle_number represents a percent of the remaining
    #particles - we then multiply by particle_number to find the actual number
    #of effective particles    
    n_out = around(weights*kill_index*particle_number**2)
    sm = sum(n_out)    
    df = (particle_number - sm)  
    sn = sign(df)
    df = int(abs(df))
    #Randomly refills the weights up to the number of the sample size
    #for use in calculating the weighted covariance for resampling
    for ii in range(df):
        idx = randint(particle_number) 
        newnum = n_out[idx] + sn
        while ((newnum < 0) or (newnum > particle_number)):
            idx = randint(particle_number)
            newnum = n_out[idx] + sn
        n_out[idx] = newnum

    cov_weights = n_out/sum(n_out) 
    #Calculate the weighted mean and weighted covariance
    mean_state = mean((parts*cov_weights),1)*particle_number
    delta_part = (mean_state - parts.T).T*sqrt(cov_weights)
    covar = cov(delta_part)*particle_number
    
    idx=0 
    r_samp = zeros([s_dim, particle_number])
    # select n_out(ii) samples from a Gaussian with mean xin(ii) 
    # and covariance covar*cov_fac
    for ii in range(particle_number):
        if (n_out[ii] > 0):
            r_samp[:,idx:idx+n_out[ii]] = multivariate_normal(parts[:,ii], 
                                                              covar*cov_fac,
                                                              [n_out[ii]]).T
            
            idx = idx + n_out[ii]
    
    return(r_samp)
    