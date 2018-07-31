import numpy as np
from l96 import l96_rk4_step
from l96 import l96_jacobian
from ekf_riccati import ekf_fore
from observation_operators import alt_obs
from observation_operators import obs_seq
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

    [seed, model_er, obs_un, sys_dim, obs_dim] = args

    # state dimension
    #sys_dim = 10
    #obs_dim = 10

    # forcing
    f = 8

    # number of analyses
    nanl = 100

    # analysis time
    tanl = .05

    # nonlinear step size
    h = .05

    # continous time of the spin period
    spin = 50

    # burn in period for the da routine
    burn = 10

    # initial value
    np.random.seed(seed)
    x = np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * sys_dim)

    ####################################################################################################################
    # generate the truth

    # spin the model on to the attractor
    for i in range(int(spin / h)):
        x = l96_rk4_step(x, h, f)
        x += np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * model_er)

    # the model and the truth will have the same initial condition
    x_init = x

    # generate the full length of the truth trajectory which we assimilate
    truth = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        x = l96_rk4_step(x, h, f) + np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * model_er)
        truth[:, i] = x

    ####################################################################################################################
    # we simulate EKF for the set observations
    H = alt_obs(sys_dim, obs_dim)
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)

    P = np.eye(sys_dim) * model_er
    R = np.eye(obs_dim) * obs_un
    Q = np.eye(sys_dim) * model_er
    I = np.eye(sys_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # re-initialize the tangent linear model
        Y = np.eye(sys_dim)
        x = l96_rk4_step(x, h, f)
        Y = Y + h * l96_jacobian(x)

        # define the forecast state and uncertainty
        x_fore[:, i] = x
        P = ekf_fore(Y, P, Q, R, H, I)

        # define the kalman gain and perform analysis
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (obs[:, i] - H @ x)
        x_anal[:, i] = x

    # compute the rmse for the forecast and analysis
    fore = x_fore - truth
    fore = np.sqrt(np.mean(fore * fore, axis=0))

    anal = x_anal - truth
    anal = np.sqrt(np.mean(anal * anal, axis=0))

    f_rmse = []
    a_rmse = []

    for i in range(burn + 1, nanl):
        f_rmse.append(np.mean(fore[:i]))
        a_rmse.append(np.mean(anal[:i]))

    ####################################################################################################################
    # save results

    data = {'fore': f_rmse, 'anal': a_rmse}

    f_name = './data/ekf_bench_rmse_seed_'+ str(seed).zfill(3) + '_analint_' + str(tanl).zfill(3) + \
             '_sys_dim_' + str(sys_dim).zfill(2) + '_obs_dim_' + str(obs_dim).zfill(2) + \
             '_obs_un_' + str(obs_un).zfill(3) + '_model_err_' + str(model_er).zfill(3) + '.txt'
    f = open(f_name, 'wb')
    pickle.dump(data, f)
    f.close()


########################################################################################################################

experiment([0, 1, .1, 10, 10])
