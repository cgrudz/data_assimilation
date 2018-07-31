import numpy as np
from l96 import l96_rk4_step
from ekf_riccati import simulate_ekf_jump
from ekf_ause_riccati import simulate_kf_ause_jump
from ekf_ause_riccati import simulate_ekf_aus_jump
import pickle
from scipy.linalg import circulant

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

    # define the initial/ model error covariance
    Q = circulant([1, .5, .25] + [0]*(sys_dim - 5) + [.25, .5])
    Q = Q * model_er

    # initial value
    np.random.seed(seed)
    x = np.random.multivariate_normal(np.zeros(sys_dim), Q * sys_dim)

    # define the storage for the climatological covariance
    clim = np.zeros([sys_dim, int(spin / nl_step / nl_fore_steps)])

    ####################################################################################################################
    # generate the truth

    j = 0
    # spin the model on to the attractor
    for i in range(int(spin / nl_step)):
        x = l96_rk4_step(x, nl_step, f)
        if (i % nl_fore_steps) == 0:
            # add jumps to the process
            x += np.random.multivariate_normal(np.zeros(sys_dim), Q) * tanl
            clim[:, j] = x
            j += 1

    clim = np.var(clim, axis=1)
    clim = np.diag(clim)

    # the model and the truth will have the same initial condition
    x_init = x

    # generate the full length of the truth tajectory which we assimilate
    truth = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # integrate until the next observation time
        for j in range(nl_fore_steps):
            x = l96_rk4_step(x, nl_step, f)

        # jump at observation
        x += np.random.multivariate_normal(np.zeros(sys_dim), Q) * tanl
        truth[:, i] = x

    ####################################################################################################################
    # we simulate EKF-AUS and others with inflation factors
    ####################################################################################################################

    m_infl_range = np.linspace(0, 3, 31)
    a_infl_range = np.linspace(0, .5, 11)
    for i in range(2):
        # we reinitialize a storage dictionary, to store at each loop of i
        data = {}

        for j in range(2):
            m_infl = 1.0 + m_infl_range[i]
            a_infl = 0.1 + a_infl_range[j]

            key = 'd_17_aus_m_infl_' + str(m_infl).zfill(2) +'_a_infl_' + str(np.round(a_infl, decimals=2)).zfill(2)
            data[key] = simulate_ekf_aus_jump(x_init, truth, 17, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un,
                                                 seed, burn, m_inf=m_infl, a_inf=a_infl, clim=clim)
            # keep track of how many completed
            print(i + j)

        # on each loop of multiplicative inflation, we store the results
        f_name = './data/AUS_total_inflation_rmse_seed_' + str(seed).zfill(3) + '_analint_' + str(tanl).zfill(3) + \
                 '_obs_un_' + str(obs_un).zfill(3) + '_model_err_' + str(model_er).zfill(3) + '_sys_dim_' + \
                 str(sys_dim).zfill(2) + '_obs_dim_' + str(obs_dim) + '_m_infl_' + \
                 str(np.round(m_infl, decimals=2)).zfill(2) + '.txt'

        fid = open(f_name, 'wb')
        pickle.dump(data, fid)
        fid.close()

        # key = 'ekf_infl_' + str(i).zfill(2)
        # data[key] = simulate_ekf_jump(x_init, truth, obs_dim, h, f, tanl, tl_fore_steps, model_er, obs_un,
        #                               seed, burn, infl=inflation)

        # key = 'd_17_kfause_infl_' + str(i).zfill(2)
        # data[key] = simulate_kf_ause_jump(x_init, truth, 17, obs_dim, h, f, tanl, tl_fore_steps, model_er, obs_un,
        #                           seed, burn, infl=inflation)



########################################################################################################################

experiment([0, .1, .5, .25, 40])
