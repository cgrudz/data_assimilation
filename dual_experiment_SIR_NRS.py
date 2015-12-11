###############################################################################################################################
import numpy as np
import pickle
import random
from ikeda_V import ikeda as model
from scipy.integrate import odeint
from no_resample_discrete_filter import NRS_filter as NRS
from SIR_discrete_filter import SIR_filter as SIR
from generate_observations_discrete import gen_obs
from distribution_plt_SIR import distribution_plt_SIR as plt_SIR
from distribution_plt_NRS import distribution_plt_NRS as plt_NRS

###############################################################################################################################
## System Parameters

# Define the number of particles in the ensemble
particle_number = int(1e3)

# set random seeds for observations and initializations of priors etc.
random.seed()
obs_seed = random.getstate()

random.setstate(obs_seed)
random.jumpahead(1000)
seed_1 = random.getstate()
random.setstate(obs_seed)
random.jumpahead(10000)
seed_2 = random.getstate()

# experiment name
name = 'dual_experiment_' + str(particle_number) + '_part' 
exp_name = name + '.p'
save_file = open('./experiments/' + exp_name,'wb')
directory = './experiments/' + name + '/'

# state dimension
state_dim = 2

# observation dimension
obs_dim = 2

# observation operator
H = np.eye(state_dim)

# weight threshold
W_thresh = 1e-10
# Neff threshold
N_thresh = .25*float(particle_number)
# resampler tuning parameter
tuning = .05

# Obs Err variance (% of climate variance) 
obs_var = 0.1

# prior covariance
p_cov = np.eye(state_dim)*.1

# Analysis performed after tanl steps
tanl = 1

# Number of Analyses (after the analysis at time zero)
nanl = 50

# Experiment length defined
exp_len = tanl*nanl

parameters = {'obs_seed':obs_seed, 'obs_var': obs_var, 'seed_1':seed_1, 'seed_2': seed_2, 
              'prior_cov':p_cov, 'W_threshold':W_thresh,
              'N_thresh':N_thresh,'tuning':tuning}

###############################################################################################################################
## Initial conditions

# Define the initial condition for the truth
truth = np.array([.5,0])

# propagate the truth for the length of the model, and return this trajector and the
# noisy observations of the state
random.setstate(obs_seed)
[truth_traj,obs,Q] = gen_obs(model,truth,H,exp_len,nanl,tanl,obs_var)

# define the priors by Gaussian with mean at the initial true state and specified covariance
P = np.eye(2)*.1

random.setstate(seed_1)
prior_1 = np.random.multivariate_normal(truth,P,particle_number)
    

random.setstate(seed_2)
prior_2 = np.random.multivariate_normal(truth,P,particle_number)

# check the priors for consistency
mean_1 = np.mean(prior_1,axis=0)
mean_2 = np.mean(prior_2,axis=0)

prior_mean_diff = mean_1 - mean_2
prior_mean_diff = np.sqrt(prior_mean_diff.dot(prior_mean_diff))

###############################################################################################################################
## No resample particle filter step

# Note that aside from the prior, there is no other randomness in this run

# initialize the filter with prior from seed 1
NRS_pdf_series_1 = NRS(model,prior_1,state_dim,particle_number,nanl,tanl,obs,Q,W_thresh)

# initialize the filter with prior from seed 2
NRS_pdf_series_2 = NRS(model,prior_2,state_dim,particle_number,nanl,tanl,obs,Q,W_thresh)

###############################################################################################################################
## SIR filter step

# initialize the filter with seed 1
random.setstate(seed_1)
SIR_pdf_series_1 = SIR(model,prior_1,state_dim,particle_number,nanl,tanl,obs,Q,N_thresh,tuning)

# initialize the filter with seed 2
random.setstate(seed_2)
SIR_pdf_series_2 = SIR(model,prior_2,state_dim,particle_number,nanl,tanl,obs,Q,N_thresh,tuning)

###############################################################################################################################
## plot and pickle data

# plot the no resample runs
plt_NRS(NRS_pdf_series_1,obs,directory,1)
plt_NRS(NRS_pdf_series_2,obs,directory,2)

# plot the SIR runs
plt_SIR(SIR_pdf_series_1,obs,directory,1)
plt_SIR(SIR_pdf_series_2,obs,directory,2)

experiment_data = {'SIR_pdf_series_1': SIR_pdf_series_1, 'SIR_pdf_series_2': SIR_pdf_series_2,
                   'NRS_pdf_series_1': NRS_pdf_series_1, 'NRS_pdf_series_2': NRS_pdf_series_2,
                   'observations': obs, 'parameters': parameters,'prior_mean_diff': prior_mean_diff}


#pickle data
pickle.dump(experiment_data, save_file)
save_file.close()

###############################################################################################################################