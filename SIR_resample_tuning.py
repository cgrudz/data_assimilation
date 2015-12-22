import numpy as np
from resample_systematic import resample_systematic as RS
from pylab import find
import pickle

# load the distribution data
exp = pickle.load(open('./experiments/dual_experiment_1000_part.p'))
exp['A_17'].keys

cov = np.cov(cloud.transpose())
resampled = RS(weights)
weights = np.ones(ens_size)/float(ens_size)
cloud = cloud[resampled.astype(int),:] + np.random.multivariate_normal([0,0],tuning*cov,ens_size)