###############################################################################
## Particle Filter Driver
###############################################################################

from pylab import *
from L63_vectorized import L63_vectorized as model
from particle_filter import bootstrap
from scipy.integrate import odeint

###############################################################################
## System Parameters
###############################################################################

# Define the number of particles in the ensemble
root   = 1
particle_number = root**3

# Spin time
spin_end = 10

# Time step
dt = .01

# Spin interval
spin = linspace(0,spin_end,spin_end/dt)

# Obs Err variance (% of climate variance) 
obs_var = 0.01

# Analysis interval
tanl = 100

# Number of Analyses
nanl = 100

# Experiment length
exp_len = tanl*nanl

# state dimension
state_dim = 3


###############################################################################
## Ensemble Spin Up
###############################################################################

#Assign cubes of initial conditions for the ensemble spin up, run the spin,
#calculate spin mean, and variance to set initial conditions for the runs
ens_base = randint(-100,100)

#Define cube of initial conditions for the spin up 
cube = zeros([state_dim,particle_number])
    
for i in range(root):
    for j in range(root):
        for k in range(root):            
            cube[:,(root**2)*i + root*j + k] =(array([k*exp(-1), 
                                                      j*exp(-1),
                                                      i*exp(-1)])+ens_base)

#Spin up the initial cloud
spin_cloud = reshape(cube.transpose(),3*particle_number)
spin_cloud = odeint(model,spin_cloud,spin)

#Determine the mean for the spin cloud at each time step           
spun_cloud = reshape(spin_cloud[-1,:],[particle_number,state_dim]).transpose()


                                
spin_mean = mean(spin_cloud,axis=1)

#Calculate variance of the mean along spin up
cl_var = 2*var(spin_cloud,axis=1)

#Observational Error covariance stored    
R = obs_var*cl_var.dot(eye(state_dim))
    
#Generate random ensemble member to be initializaiton for truth
P = randint(0,particle_number)
truth = spin_cloud[-1,P]
    
#Truth state is eliminated from the particle cloud
spin_cloud = delete(spin_cloud,P,1)
    
#Create observations with error, note the tanl-1 so that accounts 
#for index beginning at 0
obs = truth[:,tanl-1:tfin:tanl] + (randn(Nanl,3)*sqrt(obs_var*cl_var)).T

###############################################################################
## Run particle filter with noisy observations with the spun ensemble
###############################################################################

