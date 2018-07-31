import numpy as np
from l96 import l96_rk4_step
from l96 import l96_tl_fms
from lyapunov import stable_bound
from ricatti_equation import ricatti_evo
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
    sys_dim = 10

    # forcing
    f = 8

    # number of analyses
    nanl = 110

    # tlm integration step size (nonlinear step size is half)
    h = .01
    nl_step = .5 * h

    # nonlinear integration steps between observations
    tl_fore_steps = int(tanl / h)

    # continous time of the spin period
    spin = 5

    # burn in period for the da routine
    # burn = 10

    # QR algorithm convergence lead steps
    lv_lead = 100

    # define the initial error covariance
    P_init = np.eye(sys_dim)

    # initial value
    np.random.seed(seed)
    x = np.random.multivariate_normal(np.zeros(sys_dim), P_init * sys_dim)

    ####################################################################################################################
    # generate the truth

    # spin the model on to the attractor
    for i in range(int(spin / nl_step)):
        x = l96_rk4_step(x, nl_step, f)

    ####################################################################################################################
    # define the linear model propagators and observation operators:
    # generate the sequence of fundamental matrix solutions, compute the BLVs over the analysis interval

    perts = np.eye(sys_dim)
    for i in range(lv_lead):
        [fms, x] = l96_tl_fms(x, tl_fore_steps, h, 1, f, l96_rk4_step)
        perts = np.squeeze(fms) @ perts
        [perts, R] = np.linalg.qr(perts, mode='complete')

    # compute the fms over the analysis times
    R_hist = np.zeros([sys_dim, sys_dim, nanl])
    LLEs = np.zeros([sys_dim, nanl])

    # define storage for observation operators
    H_triv = np.eye(sys_dim, M=obs_dim)
    H_r = np.zeros([sys_dim, obs_dim, nanl])
    H_fixed = np.zeros([sys_dim, obs_dim, nanl])
    H_full = np.zeros([sys_dim, sys_dim, nanl])
    H_b = np.zeros([sys_dim, obs_dim, nanl])
    H_f = np.zeros([sys_dim, obs_dim, nanl])
    M_kl = np.zeros([sys_dim, sys_dim, nanl])

    for i in range(nanl):
        # define the observation operators at times 0 through nanl-1
        H_b[:, :, i] = perts[:, :obs_dim]
        H_fixed[:, :, i] = H_triv
        H_full[:, :, i] = np.eye(sys_dim)
        tmp = np.random.multivariate_normal(np.zeros(sys_dim), P_init * sys_dim, obs_dim)
        tmp, foo = np.linalg.qr(tmp.T)
        H_r[:, :, i] = tmp

        # propagate the forward model
        [fms, x] = l96_tl_fms(x, tl_fore_steps, h, 1, f, l96_rk4_step)
        M_kl[:, :, i] = np.squeeze(fms)

        # QR step to define the next BLVs and the R matrix
        perts = np.squeeze(fms).dot(perts)
        [perts, R] = np.linalg.qr(perts)
        R_hist[:, :, i] = R
        R_diag = np.log(np.abs(np.diagonal(R)))
        LLEs[:, i] = R_diag[:]

    bnd = stable_bound(R_hist[4:, 4:, :])

    # compute the forward Lyapunov vectors in batches, with a forward lead for convergence over the M_kl interval
    batches = 4
    batch_lead = int(lv_lead / batches)

    flv_batches = {}
    for i in range(batches):
        [fms, x] = l96_tl_fms(x, tl_fore_steps, h, batch_lead, f, l96_rk4_step)
        key_name = 'b_' + str(i).zfill(2)

        # note that the fundamental matrix solutions are stored in reverse order for the QR factorizations of the
        # inverse adjoint
        flv_batches[key_name] = fms[:, :, ::-1]

    # we reverse sort the keys
    keys = sorted(flv_batches.keys())[::-1]
    perts = np.eye(sys_dim)

    for key in keys:
        # and unpack the tangent linear model solutions, in reverse order
        fms = flv_batches[key]

        for i in range(batch_lead):
            perts = np.squeeze(fms[:, :, i]).T @ perts
            [perts, R] = np.linalg.qr(perts)

    del flv_batches

    # from the spun up FLVs, we compute the FLV observation operator over the analysis interval; note, we end with
    # the FLVs at zero
    for i in range(nanl)[::-1]:
        perts = np.squeeze(M_kl[:, :, i]).T @ perts
        [perts, R] = np.linalg.qr(perts)
        H_f[:, :, i] = perts[:, :obs_dim]

    ####################################################################################################################
    # compute the ricatti equation for each experimental set up
    P_b = ricatti_evo(M_kl, P_init, model_er, obs_un, H_b)
    P_f = ricatti_evo(M_kl, P_init, model_er, obs_un, H_f)
    P_full = ricatti_evo(M_kl, P_init, model_er, obs_un, H_full)
    P_r = ricatti_evo(M_kl, P_init, model_er, obs_un, H_r)
    P_fixed = ricatti_evo(M_kl, P_init, model_er, obs_un, H_fixed)

    ####################################################################################################################
    # save results

    # define dictionary for storage
    exp_data = {'P_b': P_b, 'P_f': P_f, 'P_full': P_full, 'P_random': P_r, 'P_fixed': P_fixed, 'lles': LLEs,
                'obs_err': obs_un, 'model_err': model_er, 'bnd': bnd,
                }

    # save the data
    file_name = './KF_obs_data/' + 'obs_eigs_exp' + '_f_' + str(f).zfill(3) + '_tanl_' + str(tanl).zfill(3) + \
                '_obs_dim_' + str(obs_dim).zfill(2) + '.txt'
    f = open(file_name, 'wb')
    pickle.dump(exp_data, f)
    f.close()

    return args

########################################################################################################################


experiment([0, 0.1, 1, 1, 4])
