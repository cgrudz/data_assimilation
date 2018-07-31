import numpy as np
from l96 import l96_rk4_step
from l96 import l96_step_TLM
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

    [seed, tanl] = args

    # state dimension
    sys_dim = 40

    # forcing
    f = 8

    # number of analyses
    nanl = 12

    # storage for lles
    lle = np.zeros([sys_dim, nanl])


    # tlm integration step size (nonlinear step size is half)
    h = .01
    nl_step = .5 * h

    # nonlinear integration steps between observations
    tl_fore_steps = int(tanl / h)
    nl_fore_steps = int(tanl / nl_step)

    # continous time of the spin period
    spin = 50

    # initial value
    np.random.seed(seed)
    x = np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * sys_dim)

    # QR lead time to convergence of the BLVs in terms of the number of analysis times
    lead = 50

    ####################################################################################################################
    # Spin up non-linear model onto the attractor

    for i in range(int(spin * nl_fore_steps)):
        x = l96_rk4_step(x, nl_step, f)

    ####################################################################################################################
    # propagate the tangent linear model and compute the lles

    # initialize and spin to convergence
    B = np.eye(sys_dim)

    for i in range(lead):
        # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
        # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model

        for j in range(tl_fore_steps):
            [x, B] = l96_step_TLM(x, B, h, f, l96_rk4_step)


        # compute the backward Lyapunov vectors at the next step recursively for each number of lead steps
        B, T = np.linalg.qr(B)

    # compute the lles and store for each analysis time
    for i in range(nanl):
        # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
        # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model

        for j in range(tl_fore_steps):
            [x, B] = l96_step_TLM(x, B, h, f, l96_rk4_step)

        # compute the backward Lyapunov vectors at the next step recursively
        B, T = np.linalg.qr(B)

        # compute the log of the absolute value, normalized by the interval over which the value is computed
        lle[:, i] = np.log(np.abs(np.diagonal(T))) / tanl

    ####################################################################################################################
    # save results

    data = {'lle_dist': lle}

    f_name = './data/lle_dist_seed_' + str(seed).zfill(3) + '_analint_' + str(tanl).zfill(3) + '_sys_dim_' + \
             str(sys_dim).zfill(2) + '.txt'
    f = open(f_name, 'wb')
    pickle.dump(data, f)
    f.close()

########################################################################################################################

experiment([0, .1])
