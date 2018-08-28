import numpy as np
from l96 import l96_tl_fms
from l96 import l96_rk4_step
from lyapunov import l_back
from lyapunov import l_for
from lyapunov import l_covs_range
from lyapunov import stable_bound
from kf_riccati import simulate_kf
from observation_operators import id_obs
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

    # unpack the seed and error stats
    [j, obs_un, model_err] = args

    # forcing parameter
    f = 8

    # system dimension
    sys_dim = 10

    # batch lead time --- we use multiple batches to compute the convergence of Lyapunov vectors
    batch_lead = 50

    # number of batches to compute the convergence of LVs
    no_batch = 2

    # number of analyses
    nanl = 100

    # number of integration steps between analyses in TLM
    obs_steps = 10

    # scale of observational uncertainty
    # obs_un = 1

    # scale of model error
    # model_err = 1

    # tlm integration step size (underlying step size is half)
    h = .01

    # length of continuous interval between observations
    obs_int = obs_steps*h

    # initial spin up period for the underlying non-linear solution
    nl_step = .5*h
    steps = 50

    ####################################################################################################################
    # Spin up non-linear model onto the attractor

    # randomly initialize the experiment and spin onto the attractor
    np.random.seed(j)
    x = np.random.rand(sys_dim) * sys_dim

    for i in range(steps):
        x = l96_rk4_step(x, nl_step, f)

    ####################################################################################################################
    # define the linear model propagators and observation operators:
    # generate the sequence of fundamental matrix solutions in batches, compute the BLVs/FLVs over the analysis interval
    # the analysis interval is padded forward and backward with solutions to the TLM so that we can compute the forward
    # and backward Lyapunov vectors to convergence over the extended interval

    # first batch of QR factors computed
    [back_lead_1, x] = l96_tl_fms(x, obs_steps, h, batch_lead, f)

    # we compute the local lyapunov exponents, the sum of all lles, and the backwards vectors at the end of the first
    # batch; these BLVs are used to initialize the next batch, and we take average exponents over all batches
    [perts, b_expos] = l_back(back_lead_1, np.eye(sys_dim), np.zeros(sys_dim))

    del back_lead_1
    [back_lead_2, x] = l96_tl_fms(x, obs_steps, h, batch_lead, f)

    # second batch
    [perts, b_expos] = l_back(back_lead_2, perts, b_expos)
    del back_lead_2

    # asymptotic exponents calculated over the spin period
    b_expos = b_expos / (batch_lead * no_batch)

    # compute the fms over the analysis times
    [analysis_int, x] = l96_tl_fms(x, obs_steps, h, nanl, f)

    # find the associated BLVs over the analysis times
    [perts, temp, T_kl, BLVs] = l_back(analysis_int, perts, np.zeros(sys_dim), LV_hist=True, R_trans_hist=True)

    # repeat batch process with the forward vectors
    [for_lead_1, x] = l96_tl_fms(x, obs_steps, h, batch_lead, f)

    # generate next fms batch
    [for_lead_2, x] = l96_tl_fms(x, obs_steps, h, batch_lead, f)

    # generate forward lyapunov vectors for the analysis times, moving from end time to start
    [perts, f_expos] = l_for(for_lead_2, np.eye(sys_dim), np.zeros(sys_dim))
    [perts, f_expos] = l_for(for_lead_1, perts, f_expos)

    f_expos = f_expos / (batch_lead * no_batch)

    del for_lead_1
    del for_lead_2

    [perts, tmp, FLVs] = l_for(analysis_int, perts, np.zeros(sys_dim), LV_hist=True)

    # with the BLVs and FLVs run to convergence over the same interval, we compute the CLVs via the method of LU
    CLVs = l_covs_range(BLVs, FLVs)

    ####################################################################################################################
    # set the experiment propagator and initial condition, we vary over the observational dimension, operator, and the
    # forms of the full and reduced rank filters
    M_kl = analysis_int

    # NOTE: WE ALWAYS ASSUME P_0 IS EQUIVALENT TO THE MODEL ERROR
    P_init = np.eye(sys_dim) * obs_un

    ####################################################################################################################
    # full rank filters
    # un obs
    kf_4 = simulate_kf(FLVs, BLVs, CLVs, sys_dim, 4,  M_kl, model_err, obs_un, nanl, j)
    # ws obs
    kf_5 = simulate_kf(FLVs, BLVs, CLVs, sys_dim, 5,  M_kl, model_err, obs_un, nanl, j)
    # s obs
    kf_6 = simulate_kf(FLVs, BLVs, CLVs, sys_dim, 6, M_kl, model_err, obs_un, nanl, j)
    # s obs
    kf_7 = simulate_kf(FLVs, BLVs, CLVs, sys_dim, 7, M_kl, model_err, obs_un, nanl, j)
    # s obs
    kf_8 = simulate_kf(FLVs, BLVs, CLVs, sys_dim, 8, M_kl, model_err, obs_un, nanl, j)
    # s obs
    kf_9 = simulate_kf(FLVs, BLVs, CLVs, sys_dim, 9, M_kl, model_err, obs_un, nanl, j)
    # s obs
    kf_10 = simulate_kf(FLVs, BLVs, CLVs, sys_dim, 10, M_kl, model_err, obs_un, nanl, j)

    ####################################################################################################################
    # # full dimensional obs for reference
    # p = KF_fore(M_kl, P_init, model_err, obs_un, id_obs(sys_dim, sys_dim, nanl) / np.sqrt(sys_dim))
    # p_I = []

    # for i in range(nanl):
    #     tmp_p = np.linalg.eigvalsh(np.squeeze(p[:, :, i+1]))
    #     p_I.append(tmp_p[::-1])
    #
    # kf_full_obs = np.array(p_I)

    ####################################################################################################################
    # # reduced rank filters
    #
    # # un obs subspace
    # AUS_4 = simulate_kfAUS(T_kl, 4, model_err, obs_un)
    #
    # # un/ws obs subspace
    # AUS_5 = simulate_kfAUS(T_kl, 5, model_err, obs_un)
    #
    # # un/s obs subspace
    # AUS_6 = simulate_kfAUS(T_kl, 6, model_err, obs_un)
    #
    # # un/s obs subspace
    # AUS_7 = simulate_kfAUS(T_kl, 7, model_err, obs_un)
    #
    # # un/s obs subspace
    # AUS_8 = simulate_kfAUS(T_kl, 8, model_err, obs_un)
    #
    # # un/s obs subspace
    # AUS_9 = simulate_kfAUS(T_kl, 9, model_err, obs_un)

    ####################################################################################################################
    # save results

    # define dictionary for storage
    exp_data = {'b_expos': b_expos, 'f_expos': f_expos,
                'obs_int': obs_int, 'obs_err': obs_un, 'model_err': model_err,
                'kf_4': kf_4, 'kf_5': kf_5, 'kf_6': kf_6, 'kf_7': kf_7, 'kf_8': kf_8, 'kf_9':kf_9, 'kf_10': kf_10,
                # 'kf_full_obs': kf_full_obs,
                # 'AUS_4': AUS_4, 'AUS_5': AUS_5, 'AUS_6': AUS_6, 'AUS_7': AUS_7, 'AUS_8': AUS_8, 'AUS_9': AUS_9,
                }

    # save the data
    file_name = './forecast_eigs_data/' + 'fore_eigs_seed_' + str(j).zfill(3) + '_f_' + str(f).zfill(3) \
                + '_Q_' + str(model_err).zfill(2) + '_R_' + str(obs_un).zfill(2) + '.txt'
    f = open(file_name, 'wb')
    pickle.dump(exp_data, f)
    f.close()

    return args

########################################################################################################################

experiment([300, 1, 1])
