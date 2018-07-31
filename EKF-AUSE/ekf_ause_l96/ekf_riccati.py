import numpy as np
from observation_operators import alt_obs
from observation_operators import obs_seq
from l96 import l96_2tay_step
from l96 import l96_step_TLM
from l96 import l96_rk4_step
import copy

########################################################################################################################
# forecast ricatti equation


def ekf_fore(M, P, Q, R, H, I, inflation=1.0):
    """"This function returns a sequence of analysis error covariance matrices for the ricatti equation"""

    # the forecast riccati is given in the square root form
    [U, s, V] = np.linalg.svd(P)
    X = U @ np.diag(np.sqrt(s))

    Omega = H.T @ np.linalg.inv(R) @ H
    XOX = X.T @ Omega @ X

    P = M @ X @ np.linalg.inv(I + XOX) @ X.T @ M.T + Q
    P = P * inflation

    return P

# ########################################################################################################################
# simulate the sde model error twin experiment


def simulate_ekf_sde(x, truth, obs_dim, h, f, tl_fore_steps, diffusion, obs_un, seed, burn, infl=1.0):
    """This function simulates the extended kalman filter with alternating obs operator"""

    # define initial parameters
    [sys_dim, nanl] = np.shape(truth)

    # define the observations
    H = alt_obs(sys_dim, obs_dim)
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
    P = np.eye(sys_dim)

    R = np.eye(obs_dim) * obs_un
    I = np.eye(sys_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # re-initialize the tangent linear model
        Y = np.eye(sys_dim)
        # re-initialize the tangent linear model history for the error covariance propagation
        Y_hist = np.zeros([sys_dim, sys_dim, tl_fore_steps])

        # make a forecast along the non-linear trajectory and generate the tangent model, looping in the integration
        # steps of the tangent linear model
        for j in range(tl_fore_steps):
            # we use the 2nd order taylor scheme for nonlinear
            [x, Y] = l96_step_TLM(x, Y, h, f, l96_2tay_step)
            Y_hist[:, :, j] = Y

        # Q is defined by the forward linear evolved wiener process covariance scaled by diffusion squared
        # (we are assuming that diffusion is modelled by a scalar matrix, which is applied to the standard normal
        # weiner process).  We estimate the integral with the left Reimann sum

        # left most value
        Q = Y @ Y.T * h
        for j in range(tl_fore_steps - 1):
            # sum through all remaining values in the inverval
            PHI = Y @ np.linalg.inv(np.squeeze(Y_hist[:, :, j]))
            Q += PHI @ PHI.T * h

        Q = diffusion**2 * Q

        # define the forecast state and uncertainty
        x_fore[:, i] = x
        P = ekf_fore(Y, P, Q, R, H, I, inflation=infl)

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

    return {'fore': f_rmse, 'anal': a_rmse}

########################################################################################################################
# simulate


def simulate_ekf_jump(x, truth, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un, seed, burn, infl=1.0):
    """This function simulates the extended kalman filter with alternating obs operator"""

    # define initial parameters
    [sys_dim, nanl] = np.shape(truth)

    # define the observations
    H = alt_obs(sys_dim, obs_dim)
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
    P = np.eye(sys_dim)
    Q = Q * tanl**2
    R = np.eye(obs_dim) * obs_un
    I = np.eye(sys_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # re-initialize the tangent linear model
        Y = np.eye(sys_dim)

        # make a forecast along the non-linear trajectory and generate the tangent model, looping in the integration
        # steps of the tangent linear model
        for j in range(tl_fore_steps):
            [x, Y] = l96_step_TLM(x, Y, h, f, l96_rk4_step)

        # define the forecast state and uncertainty
        x_fore[:, i] = x
        P = ekf_fore(Y, P, Q, R, H, I, inflation=infl)

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

    return {'fore': f_rmse, 'anal': a_rmse}

