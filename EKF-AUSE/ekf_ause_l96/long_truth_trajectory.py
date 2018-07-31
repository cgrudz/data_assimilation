from l96 import l96_2tay_sde
import numpy as np
import pickle
import time
import copy


def experiment(args):
    """This experiment will spin a "truth" trajectory of the stochastic l96 and store a series of analysis points

    This is a function of the ensemble number which initializes the random seed.  This will pickle the associated
    trajectories for processing in data assimilation experiments."""

    ####################################################################################################################
    [seed, diffusion, analint] = args

    # static parameters
    f = 8
    sys_dim = 10
    p = 2
    h = .0025
    # number of observations
    nanl = 12
    spin = 5
    fore_steps = int(analint / h)

    # define the initialization of the model
    np.random.seed(seed)
    x = np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * sys_dim)

    # static parameters based on fourier approxmimation cut off
    RHO = 1 / 12 - .5 * np.pi ** (-2) * np.sum([1 / (r ** 2) for r in range(1, p + 1)])
    ALPHA = (np.pi**2) / 180 - .5 * np.pi**(-2) * np.sum([1/r**4 for r in range(1, p+1)])

    t = time.time()
    # spin is the length of the spin period in the continuous time variable
    for i in range(int(spin / h)):
        # integrate one step forward, for each
        x = l96_2tay_sde(x, h, f, diffusion, p, rho=RHO, alpha=ALPHA)

    # store initialization for the models
    x_init = copy.copy(x)

    # generate the full length of the truth tajectory which we assimilate
    truth = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # integrate until the next observation time
        for j in range(fore_steps):
            # integrate one step forward
            x = l96_2tay_sde(x, h, f, diffusion, p, rho=RHO, alpha=ALPHA)

        truth[:, i] = x

    elapsed = time.time() - t

    params = [seed, diffusion, analint, h, f, x_init]
    data = {'truth': truth, 'params': params}
    fname = './data/sde_seed_' + str(seed).zfill(3) + '_sys_dim_' + str(sys_dim).zfill(2) + '_analint_' + \
            str(analint).zfill(3) + '_diffusion_' + str(diffusion).zfill(3) + '.txt'
    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()

    return elapsed

########################################################################################################################

print(experiment([0, 1.0, 0.1]))
