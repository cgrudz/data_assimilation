import numpy as np
from l96 import l96_rk4_step
from ekf_riccati import simulate_ekf_jump
from ekf_ause_riccati import simulate_kf_ause_jump
from ekf_ause_riccati import simulate_ekf_aus_jump
from ekf_ause_riccati import simulate_ekf_ause_jump
from ekf_ause_riccati import simulate_ekf_ause_alt_jump
import pickle

########################################################################################################################
# functionalized experiment, will generate ensemble random seed for the initialization by the ensemble number and
# save the data to the associated ensemble number


def experiment(args):
    """This experiment will initialize the l96 TLM and produce an analysis of the KF against different obs operators

    This is a function of the ensemble number which initializes the random seed.  This will pickle the analysis produced
    via the KF over a random initialization of the propagator model, its associated Lyapunov projector observations,
    and a random state for the linear model."""

    ####################################################################################################################
    # Define experiment parameters

    [seed, tanl, model_er, obs_un, obs_dim] = args

    # state dimension
    sys_dim = 40

    # forcing
    f = 8

    # number of analyses
    nanl = 12

    # tlm integration step size (nonlinear step size is half)
    h = .01
    nl_step = .5 * h

    # nonlinear integration steps between observations
    tl_fore_steps = int(tanl / h)
    nl_fore_steps = int(tanl / nl_step)

    # continous time of the spin period
    spin = 50

    # burn in period for the da routine
    burn = 2

    # initial value
    np.random.seed(seed)
    x = np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * sys_dim)

    ####################################################################################################################
    # generate the truth

    # spin the model on to the attractor
    for i in range(int(spin / nl_step)):
        x = l96_rk4_step(x, nl_step, f)
        if (i % nl_fore_steps) == 0:
            # add jumps to the process
            x += np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * model_er) * tanl

    # the model and the truth will have the same initial condition
    x_init = x

    # generate the full length of the truth tajectory which we assimilate
    truth = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # integrate until the next observation time
        for j in range(nl_fore_steps):
            x = l96_rk4_step(x, nl_step, f)

        # jump at observation
        x += np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * model_er) * tanl
        truth[:, i] = x

    ####################################################################################################################
    # we simulate EKF-AUS
    d_17_kfause = simulate_kf_ause_jump(x_init, truth, 17, obs_dim, h, f, tanl, tl_fore_steps, model_er, obs_un, seed, burn)

    ####################################################################################################################
    # save results

    data = {'d_17_kfause': d_17_kfause}

    f_name = './data/ause_inflation_seed_' + str(seed).zfill(3) + '_analint_' + str(tanl).zfill(3) + '_obs_un_' +\
             str(obs_un).zfill(3) + '_model_err_' + str(model_er).zfill(3) + '_sys_dim_' + str(sys_dim).zfill(2) + \
             '_obs_dim_' + str(obs_dim) + '.txt'
    f = open(f_name, 'wb')
    pickle.dump(data, f)
    f.close()

########################################################################################################################

experiment([0, .1, 1, .25, 40])
