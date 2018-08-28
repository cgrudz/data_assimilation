import numpy as np
from l96 import l96_tl_fms
from l96 import l96_rk4_step
from lyapunov import l_back
from kf_riccati import kf_fore
from kf_riccati import analyze_p
from kf_ause_riccati import simulate_kfAUS
from observation_operators import alt_obs
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
    [seed, obs_un, model_err] = args

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
    np.random.seed(seed)
    x = np.random.rand(sys_dim) * sys_dim

    for i in range(steps):
        x = l96_rk4_step(x, nl_step, f)

    ####################################################################################################################
    # define the linear model propagators and observation operators:
    # generate the sequence of fundamental matrix solutions in batches, compute the BLVs over the analysis interval
    # the analysis interval is padded backward with solutions to the TLM so that we can compute the
    # backward Lyapunov vectors to convergence over the extended interval

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
    [perts, temp, T_kl, BLVs] = l_back(analysis_int, perts, np.zeros(sys_dim), R_trans_hist=True, LV_hist=True)

    ####################################################################################################################
    # define the observation operator
    H_4 = alt_obs(sys_dim, 4, nanl)
    H_10 = alt_obs(sys_dim, 10, nanl)

    ####################################################################################################################
    # full rank filters
    P_init = np.eye(sys_dim) * model_err

    P = kf_fore(analysis_int, P_init, model_err, obs_un, H_4)
    p_eigs = []
    p_bb = []

    # we find all eigenvalues and find the inner product with the BLVS, FLVs and CLVS
    for i in range(nanl):
        tmp_p = np.linalg.eigvalsh(np.squeeze(P[:, :, i + 1]))
        p_eigs.append(tmp_p[::-1])
        tmp_b = []

        for j in range(sys_dim):
            b_j = np.squeeze(BLVs[:, j, i])
            tmp_b.append(b_j.T @ np.squeeze(P[:, :, i + 1]) @ b_j)

        p_bb.append(tmp_b)

    KF_4 = [p_eigs, p_bb]

    P = kf_fore(analysis_int, P_init, model_err, obs_un, H_10)
    p_eigs = []
    p_bb = []

    # we find all eigenvalues and find the inner product with the BLVS, FLVs and CLVS
    for i in range(nanl):
        tmp_p = np.linalg.eigvalsh(np.squeeze(P[:, :, i + 1]))
        p_eigs.append(tmp_p[::-1])
        tmp_b = []

        for j in range(sys_dim):
            b_j = np.squeeze(BLVs[:, j, i])
            tmp_b.append(b_j.T @ np.squeeze(P[:, :, i + 1]) @ b_j)

        p_bb.append(tmp_b)

    KF_10 = [p_eigs, p_bb]

    ####################################################################################################################
    # reduced rank filters
    # un obs subspace
    AUS_4_4 = simulate_kfAUS(BLVs, T_kl, H_4, 4, model_err, obs_un)

    # un/ws obs subspace
    AUS_5_4 = simulate_kfAUS(BLVs, T_kl, H_4, 5, model_err, obs_un)

    # un/s obs subspace
    AUS_6_4 = simulate_kfAUS(BLVs, T_kl, H_4, 6, model_err, obs_un)

    # un/s obs subspace
    AUS_7_4 = simulate_kfAUS(BLVs, T_kl, H_4, 7, model_err, obs_un)

    # un/s obs subspace
    AUS_8_4 = simulate_kfAUS(BLVs, T_kl, H_4, 8, model_err, obs_un)

    # un/s obs subspace
    AUS_9_4 = simulate_kfAUS(BLVs, T_kl, H_4, 9, model_err, obs_un)

    # un obs subspace
    AUS_4_10 = simulate_kfAUS(BLVs, T_kl, H_10, 4, model_err, obs_un)

    # un/ws obs subspace
    AUS_5_10 = simulate_kfAUS(BLVs, T_kl, H_10, 5, model_err, obs_un)

    # un/s obs subspace
    AUS_6_10 = simulate_kfAUS(BLVs, T_kl, H_10, 6, model_err, obs_un)

    # un/s obs subspace
    AUS_7_10 = simulate_kfAUS(BLVs, T_kl, H_10, 7, model_err, obs_un)

    # un/s obs subspace
    AUS_8_10 = simulate_kfAUS(BLVs, T_kl, H_10, 8, model_err, obs_un)

    # un/s obs subspace
    AUS_9_10 = simulate_kfAUS(BLVs, T_kl, H_10, 9, model_err, obs_un)

    ####################################################################################################################
    # save results

    # define dictionary for storage
    exp_data = {'b_expos': b_expos, 'KF_4': KF_4, 'KF_10': KF_10,
                'obs_int': obs_int, 'obs_err': obs_un, 'model_err': model_err,
                'AUS_4_4': AUS_4_4, 'AUS_5_4': AUS_5_4, 'AUS_6_6': AUS_6_4, 'AUS_7_4': AUS_7_4, 'AUS_8_4': AUS_8_4,
                'AUS_9_4': AUS_9_4,
                'AUS_4_10': AUS_4_10, 'AUS_5_10': AUS_5_10, 'AUS_6_10': AUS_6_10, 'AUS_7_10': AUS_7_10,
                'AUS_8_10': AUS_8_10, 'AUS_9_10': AUS_9_10,
                }

    # save the data
    file_name = './forecast_eigs_data/' + 'kf_v_AUSE_eigs_seed_' + str(seed).zfill(3) \
                + '_Q_' + str(model_err).zfill(2) + '_R_' + str(obs_un).zfill(2) + '.txt'
    f = open(file_name, 'wb')
    pickle.dump(exp_data, f)
    f.close()

    return args

########################################################################################################################

experiment([300, 1, 1])
