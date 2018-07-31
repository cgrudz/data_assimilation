from ipyparallel import Client
import sys
import numpy as np

########################################################################################################################
# set up parallel client

rc = Client()
dview = rc[:]

with dview.sync_imports():
    from kf_obs_cov_eigs_exp import experiment



exps = []
seed=0
tanl=0.1
model_er=1
obs_un=1
for i in range(4,11):
    exps.append([seed, tanl, model_er, obs_un, i])

# run the experiments given the parameters and write to text files, in parallel over the initializations

completed = dview.map_sync(experiment, exps)

print(completed)

sys.exit()


########################################################################################################################
