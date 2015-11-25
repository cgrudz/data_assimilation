###Lorenz 63 Particle Filter Diagnostics Module ###
from pylab import *
from PF_ikeda_fun import PF as PF
from PF_AUS_ikeda_fun import PF_AUS as PF_AUS
from ikeda_map_v import IMS
from ikeda_map_v import IMV
from mpl_toolkits.mplot3d import Axes3D

# Spin up time (Transient) in timestep
ensemble_number = 1
ens_base = 0
tspin =1
particle_number = 100
spin_mean = zeros([2,tspin+1])

# EXPERIMENT DURATION
tfin = 4000

# ASSIMILATION INTERVAL (timesteps)
tanl = 20
baus_scale = .01

# NUMBER OF ANALYSES
Nanl = int(tfin/tanl)

#Define ensemble mean arrays
mean_error_1 = zeros([ensemble_number,3,tfin+1])
avg_mean_error_1 = zeros([ensemble_number,tfin+1])
mean_error_2 = zeros([ensemble_number,3,tfin+1])
avg_mean_error_2 = zeros([ensemble_number,tfin+1])


# Obs Err variance (% of climate variance) 
obs_var = 0.01
#Climatalogical variance per ensemble
cl_var = zeros(2)

#Obs Err Covariance per ensemble
R = zeros([2,2])

# system chaotic for u = .9
u=0.9

#Assign cubes of initial conditions for the ensemble spin ups, run the spin,
#calculate spin mean, and variance to set initial conditions for the runs
for j in range(ensemble_number):
    ens_base = randint(-100,100)
    #Define the initial conditions for the spin up 
    spin_cloud = zeros([2,particle_number])

    for ii in range(10):
        for jj in range(10):
                spin_cloud[:,10*ii + jj] = (array([ii*exp(-10)/10.0,
                                                          jj*exp(-10)/10.0]) 
                                                          + ens_base)

    #Spin up the initial cloud
    spin_cloud = IMV(tspin,spin_cloud,u)
                                             
    #Determine the mean for the spin cloud at each time step           
    spin_mean = array([mean(spin_cloud[0,:,:],1),     
                       mean(spin_cloud[1,:,:],1)])
    #Calculate variance of the mean along spin up
    #cl_var = array([2*var(spin_mean[0,:]),
    #                2*var(spin_mean[1,:])])
    #Observational Error covariance stored    
    #for k in range(2):
    #    R[k,k] = obs_var*cl_var[k]
    R[0,0] = .1
    R[1,1] = .1
    
    #bred cycle perturbations created
    bred_perts = multivariate_normal([0,0],R,Nanl)
    #Generate random ensemble member to be initializaiton for truth
    P = randint(0,particle_number)
    truth = spin_cloud[:,-1,P]
    
    #Propagate spin ensemble mean forward as truth state
    truth = IMS(tfin,truth,u)
    
    #the bred trajectory and Lyapunov mode storage vectors defined
    truth_b = concatenate((spin_cloud[:,-70:-1,P],truth),axis=1)
    b_step = zeros([2,2])
    L = zeros([2,Nanl])

    #A bred vector cloud is created and propagated along the trajectory
    for k in range(Nanl):
        #Calculate the bred mode with 70 steps up to each analysis step
        bred_traj = truth_b[k*Nanl:k*Nanl+70]
        bred_cycle = len(bred_traj)
        b_cloud = bred_perts[k,:]
        for i in range(bred_cycle):    
        #Scale the pertubation vector and add it to the base point
            b_cloud = ((b_cloud/sqrt(b_cloud.dot(b_cloud)))*
                        baus_scale+bred_traj[:,i])
            #Forward propagate the perturbation            
            b_step[:,:] = IMS(1,b_cloud,u)
            #Take the difference of the forward perturbation and the forward
            #base point, 
            b_cloud = b_step[:,1] - bred_traj[:,i+1]
        #Normalized bred mode is stored
        L[:,k] = b_cloud/sqrt(b_cloud.dot(b_cloud))    
    
    #Truth state is eliminated from the particle cloud
    spin_cloud = delete(spin_cloud,P,2)
    
    #Create observations with error, note the tanl-1 so that accounts 
    #for python index beginning at 0
    obs = truth[:,tanl-1:tfin:tanl] + (randn(Nanl,2)*sqrt(obs_var*cl_var)).T
  
    #Run the PF-AUS filter from the spun ensemble with truth trajectory,
    # observations, and obs_error as defined above.  The mean
    #error sequence averaged over time for each ensemble indexed by j 
    
    [mean_state_1,mean_error_1,
     avg_mean_error_1[j,:]] = PF(truth,obs,R,spin_cloud[:,-1,:],u,
                                          tfin=4000)
    [mean_state_2,mean_error_2,
     avg_mean_error_2[j,:]] = PF_AUS(truth,obs,R,spin_cloud[:,-1,:],
                                     bred_perts.T,u,tfin=4000)
                                     
#Average the cumulative average error for each run over the all ensembles for
#the diagnostics
ensemble_avg_error_1 = mean(avg_mean_error_1,0)
ensemble_avg_error_2 = mean(avg_mean_error_2,0)
                                                   
#Plot the average cumulative average error over the ensembles against the
#experiment length

"""figure(2)
plot(range(tfin+1),x_error_B,'b--')
plot(range(tfin+1),x_error_C,'r--')

figure(3)
plot(range(tfin+1),y_error_B,'b--')
plot(range(tfin+1),y_error_C,'r--')

figure(4)
plot(range(tfin+1),z_error_B,'b--')
plot(range(tfin+1),z_error_C,'r--')"""

figure(1)
plot(range(tfin+1),ensemble_avg_error_1,'b--',label='PF')
plot(range(tfin+1),ensemble_avg_error_2,'r--',label='PF-AUS')

title('1 Avg - LS Rand -PF Ikeda ')
xlabel('Experiment Duration')
ylabel('Mean Avg Error')
legend(loc='upper right')
savefig('1_ensemble_avg_ls_rand_running_error_IM')


figure(2)
subplot(111, projection='3d')
plot(range(len(truth[0,:])),truth[0,:],truth[1,:], 'r-')
show()