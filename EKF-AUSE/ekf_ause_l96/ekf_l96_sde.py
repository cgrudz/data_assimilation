import numpy as np
from ekf_riccati import simulate_ekf_sde
import pickle

########################################################################################################################
# functionalized experiment, will generate ensemble random seed for the initialization by the ensemble number and
# save the data to the associated ensemble number


def experiment(f_name, obs_un):
    """This experiment will initialize the l96 TLM and produce an analysis of the KF against different obs operators

    This is a function of the ensemble number which initializes the random seed.  This will pickle the analysis produced
    via the KF over a random initialization of the propagator model, its associated Lyapunov projector observations,
    and a random state for the linear model."""

    ####################################################################################################################
    # Define experiment parameters

    tmp = open(f_name, 'rb')
    truth = pickle.load(tmp)
    tmp.close()

    # unpack params
    [seed, diffusion, analint, h] = truth['params']

    # truth trajectory for observations
    t_traj = truth['traj']

    # we will initialize all filters with the true state and burn in before we consider the statistics
    x_init = t_traj[:, 0]
    # re-define the truth after the initialization
    truth = t_traj[:, 1:]

    # tlm integration step size (underlying step size is half)
    tl_step = h * 2

    # forcing
    f = 8

    # number of integration steps between analyses in TLM
    tl_fore_steps = int(analint / tl_step)

    # burn in period for the da routine
    burn = 500

    ####################################################################################################################
    # we simulate EKF for the set observations

    d_4_ekf = simulate_ekf_sde(x_init, truth, 4, tl_step, f, tl_fore_steps, diffusion, obs_un, seed, burn)
    d_5_ekf = simulate_ekf_sde(x_init, truth, 5, tl_step, f, tl_fore_steps, diffusion, obs_un, seed, burn)
    d_6_ekf = simulate_ekf_sde(x_init, truth, 6, tl_step, f, tl_fore_steps, diffusion, obs_un, seed, burn)
    d_7_ekf = simulate_ekf_sde(x_init, truth, 7, tl_step, f, tl_fore_steps, diffusion, obs_un, seed, burn)
    d_8_ekf = simulate_ekf_sde(x_init, truth, 8, tl_step, f, tl_fore_steps, diffusion, obs_un, seed, burn)
    d_9_ekf = simulate_ekf_sde(x_init, truth, 9, tl_step, f, tl_fore_steps, diffusion, obs_un, seed, burn)
    d_10_ekf = simulate_ekf_sde(x_init, truth, 10,  tl_step, h, f, tl_fore_steps, diffusion, obs_un, seed, burn)


    ####################################################################################################################
    # save results

    data = {'d4': d_4_ekf, 'd5': d_5_ekf, 'd6': d_6_ekf, 'd7': d_7_ekf, 'd8': d_8_ekf, 'd9': d_9_ekf, 'd10': d_10_ekf}

    f_name = '../data/ekf_sde_rmse_seed_' + str(seed).zfill(3) + '_analint_' + str(analint).zfill(3) + '_obs_un_' +\
             str(obs_un).zfill(3) + '_diffusion_' + str(diffusion).zfill(3) + '.txt'
    f = open(f_name, 'wb')
    pickle.dump(data, f)
    f.close()


########################################################################################################################

experiment('../data/sde_seed_000_analint_0.01_diffusion_0.1.txt', .01)
