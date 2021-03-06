import numpy as np
from l96 import l96_rk4_step
from ekf_riccati import simulate_ekf_sde
from ekf_ause_riccati import simulate_kf_ause_sde
from ekf_ause_riccati import simulate_ekf_aus_sde
from scipy.linalg import circulant
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
    # Define experiment parameters, where we load the saved truth run

    [obs_dim, obs_un, fname] = args

    fid = open(fname, 'rb')
    tmp = pickle.load(fid)
    fid.close()

    [seed, diffusion, tanl, nl_h, f, x_init] = tmp['params']
    truth = tmp['truth']
    #truth = truth[:, :12]

    # state dimension and number of analyses
    [sys_dim, nanl] = np.shape(truth)

    # the tangent linear model is evolved every other time step due to the runge-kutta scheme --- we use this time step
    # in the following functions
    h = nl_h * 2

    # nonlinear integration steps between observations
    tl_fore_steps = int(tanl / h)

    # burn in period for the da routine
    burn = 2

    ####################################################################################################################
    # we simulate EKF for the set observations

    ekf = simulate_ekf_sde(x_init, truth, obs_dim, h, f, tl_fore_steps, diffusion, obs_un, seed, burn)
    data = {'ekf': ekf}
    f_name = './data/SDE_EKF_rmse_seed_' + str(seed).zfill(3) + '_analint_' + str(tanl).zfill(3) + '_obs_un_' +\
             str(obs_un).zfill(3) + '_diffusion_' + str(diffusion).zfill(3) + '_sys_dim_' + str(sys_dim).zfill(2)+\
             '_obs_dim_' + str(obs_dim) + '.txt'
    fid = open(f_name, 'wb')
    pickle.dump(data, fid)
    fid.close()

    ####################################################################################################################
    # we simulate EKF-AUS
    ####################################################################################################################
    ens_range = np.arange(4, 10)
    ens_range = ens_range[::-1]

    for i in ens_range:
        # reset dictionary
        data = {}

        # aus
        key = 'aus'
        data[key] = simulate_ekf_aus_sde(x_init, truth, i, obs_dim, h, f, tl_fore_steps, diffusion, obs_un, seed, burn)

        # then runs of kf-ause
        key = 'ause'
        data[key] = simulate_kf_ause_sde(x_init, truth, i, obs_dim, h, f, tl_fore_steps, diffusion, obs_un, seed, burn)

        # open up a storage file
        f_name = './data/SDE_AUS_AUSE_rmse_seed_' + str(seed).zfill(3) + '_analint_' + str(tanl).zfill(3) + \
                 '_obs_un_' + str(obs_un).zfill(3) + '_diffusion_' + str(diffusion).zfill(3) + '_sys_dim_' + \
                 str(sys_dim).zfill(2) + '_obs_dim_' + str(obs_dim) + '_ens_dim_' + str(i).zfill(2) + '.txt'

        fid = open(f_name, 'wb')
        pickle.dump(data, fid)
        fid.close()


########################################################################################################################

experiment([10, .25, './data/sde_seed_000_sys_dim_10_analint_0.1_diffusion_001.txt'])
