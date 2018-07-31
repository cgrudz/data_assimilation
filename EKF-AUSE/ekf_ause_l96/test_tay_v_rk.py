from l96 import l96_rk4_step
from l96 import l96_2tay_step
from l96 import l96_2tay_sde
import numpy as np
import copy
from matplotlib import pyplot as plt
import pickle

def experiment(diffusion):

    sys_dim = 10
    no_init = 500
    h = .0025
    spin = 500
    tanl = 40

    dist = np.zeros([tanl, no_init])

    for j in range(no_init):

        # take new initialization
        np.random.seed(j)
        x_init = np.random.rand(sys_dim) * sys_dim

        for i in range(int(spin/h)):
            # we spin up the solution
            x_init = l96_2tay_sde(x_init, h, 8, diffusion, 2)

            x_sd = copy.copy(x_init)
            x_ta = copy.copy(x_init)

        for i in range(tanl):

            x_ta = l96_2tay_step(x_ta, h, 8)
            x_sd = l96_2tay_sde(x_sd, h, 8, diffusion, 1)
            d = x_ta - x_sd
            dist[i, j] = np.sqrt(np.dot(d,d))

    dist = np.mean(dist, axis=1)
    data = {'mean_dist': dist}

    fname = './data/sde_divergence_diffusion_' + str(diffusion).zfill(5) + '.txt'

    fid = open(fname, 'wb')
    pickle.dump(data, fid)

########################################################################################################################
# plt.plot(np.arange(len(dist))*.0025, dist, label='Stoch v Det avg. divergence')
# # plt.plot(np.arange(len(dist2))*.05, dist1, label='Taylor v RK')
# plt.legend()
# plt.show()