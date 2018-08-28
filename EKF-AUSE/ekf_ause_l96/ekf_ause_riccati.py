import numpy as np
from observation_operators import alt_obs
from observation_operators import obs_seq
from l96 import l96_rk4_step
from l96 import l96_step_TLM
from l96 import l96_rk4_stepV
from l96 import l96_2tay_step
import copy

# ########################################################################################################################
# # augmented KFAUSE, full forecast error Riccati equation for jump equation
#
#
# def EKFAUSE_fore(P_0, B_0, T_1, H_0, Q, R):
#     """"This function the error covariance matrix for the forecast error increment of AUSE
#
#     H_f is defined to be the linearized observation operator at the forecast mean times the leading backward Lyapunov
#     vectors, i.e., the directional derivative.  We will assume Q is written in the basis of backward Lyapunov vectors
#     and is dimension of the ensemble alone. We assume that the ensemble dimension is always given as the filtered
#     dimension + 1."""
#
#     # infer parameters
#     [sys_dim, ens_dim] = np.shape(B_0)
#     f_d = ens_dim - 1
#
#     # define the filtered and unfiltered space
#     B_f = B_0[:, :f_d]
#     B_u = B_0[:, -1]
#
#     H_f = H_0 @ B_f
#     H_u = H_0 @ B_u
#
#     # define the identity matrix of dimension of the filtered space
#     I_f = np.eye(f_d)
#
#     # v = np.linalg.eigvalsh(P)
#     # if min(v) < 1:
#     #     print(v)
#
#     # restrict the ricatti equation to the filtered and unfiltered parts, note that the unfiltered is always scalar
#     S_0sqr = P_0[:f_d, :f_d]
#     P_0fu = P_0[:f_d, -1]
#     P_0uu = np.squeeze(P_0[-1, -1])
#
#     T_1ff = T_1[:f_d, :f_d]
#     T_1fu = np.reshape(T_1[:f_d, -1], [f_d, 1])
#     T_1uu = np.squeeze(T_1[-1:, -1])
#
#     # Q is assumed to already be written in a basis of backward Lyapunov vectors
#     Q_ff = Q[:f_d, :f_d]
#     Q_fu = Q[:f_d, -1]
#     Q_uu = np.squeeze(Q[-1, -1])
#
#     # define the square root of the filtered variables for the update
#     [U, S_0, V] = np.linalg.svd(S_0sqr)
#     S_0 = U @ np.diag(np.sqrt(S_0)) @ U.T
#
#     # precision matrix
#     Omega = S_0 @ H_f.T @ np.linalg.inv(R) @ H_f @ S_0
#
#     # reduced gain equation
#     J = S_0sqr @ H_f.T @ np.linalg.inv(H_f @ S_0sqr @ H_f.T + R)
#
#     # UPDATE STEPS
#
#     # filtered block update
#     S_1sqr = T_1ff @ S_0 @ np.linalg.inv(I_f + Omega) @ S_0 @ T_1ff.T + Q_ff
#     tmp = (T_1fu - np.reshape(T_1ff @ J @ H_u, [f_d, 1]))
#     S_1sqr += P_0uu * tmp @ tmp.T
#     Phi = T_1ff @ (I_f - J @ H_f) @ P_0fu
#
#     # reshape to get the exterior product of the two vectors and prevent unintended broadcasting
#     Phi = np.reshape(Phi, [f_d, 1]) @ (T_1fu - np.reshape(T_1ff @ J @ H_u, [f_d, 1])).T
#     S_1sqr += Phi + Phi.T
#
#     # unfiltered update step
#     P_1uu = Q_uu + P_0uu * T_1uu**2
#
#     # cross covariance steps
#     P_1fu = (T_1fu - np.reshape(T_1ff @ J @ H_u, [f_d, 1])) * P_0uu * T_1uu + np.reshape(Q_fu, [f_d, 1])
#     P_1fu += np.reshape(T_1ff @ (I_f - J @ H_f) @ P_0fu * T_1uu, [f_d, 1])
#     P_1fu = np.squeeze(P_1fu)
#
#     # broadcast the updates into the matrix P_1
#     P_1 = np.zeros([ens_dim, ens_dim])
#     P_1[:f_d, :f_d] = S_1sqr
#     P_1[:f_d, -1] = P_1fu
#     P_1[-1, :f_d] = P_1fu.T
#     P_1[-1, -1] = P_1uu
#
#     return P_1
#

########################################################################################################################
# augmented KFAUSE, full forecast error Riccati equation for jump equation


def EKFAUS_fore(P_0, B_0, T_1, H_0, Q_ff, R, inflation=1.0):
    """"This function the error covariance matrix for the forecast error increment of AUSE

    H_f is defined to be the linearized observation operator at the forecast mean times the leading backward Lyapunov
    vectors, i.e., the directional derivative.  We will assume Q is written in the basis of backward Lyapunov vectors
    and is dimension of the ensemble alone. We assume that the ensemble dimension is always given as the filtered
    dimension + 1."""

    # infer parameters
    [sys_dim, ens_dim] = np.shape(B_0)
    H_b = H_0 @ B_0

    # define the identity matrix of dimension of the filtered space
    I_f = np.eye(ens_dim)

    # v = np.linalg.eigvalsh(P)
    # if min(v) < 1:
    #     print(v)

    # restrict the ricatti equation to the filtered and unfiltered parts, note that the unfiltered is always scalar
    S_0sqr = P_0

    # define the square root of the filtered variables for the update
    [U, S_0, V] = np.linalg.svd(S_0sqr)
    S_0 = U @ np.diag(np.sqrt(S_0)) @ U.T

    # precision matrix
    Omega = S_0 @ H_b.T @ np.linalg.inv(R) @ H_b @ S_0

    # UPDATE STEPS

    # filtered block update
    S_1sqr = T_1 @ S_0 @ np.linalg.inv(I_f + Omega) @ S_0 @ T_1.T 
    S_1sqr = S_1sqr * inflation + Q_ff

    return S_1sqr

########################################################################################################################
# augmented KFAUSE, full forecast error Riccati equation


def KFAUSE_fore(P_0, B_0, T_1, H_0, ens_dim, Q_1, R, inflation=1.0):
    """"This function returns a sequence of error covariance matrices for the forecast error increment of KF-AUSE

    Note: we make the reduction in the calculation that R, Q are diagonal and time invariant.  T_k will be the full
    triangular matrix and BLVs will be the full basis of BLVs."""

    # infer parameters define storage and parameters for the experiment
    [obs_dim, sys_dim] = np.shape(H_0)
    unfil_d = sys_dim - ens_dim
    I_f = np.eye(ens_dim)
    I_u = np.eye(unfil_d)

    # v = np.linalg.eigvalsh(P_0)
    # if min(v) < 1:
    #     print(v)

    # break P into its sub components
    P_0uu = np.squeeze(P_0[ens_dim:, ens_dim:])
    P_0fu = P_0[:ens_dim, ens_dim:]
    S_0sqr = P_0[:ens_dim, :ens_dim]

    # define the span of the filtered and unfiltered variables
    B_0f = B_0[:, :ens_dim]
    B_0u = B_0[:, ens_dim:]

    # and the observation operator on their spans
    H_f = H_0 @ B_0f
    H_u = H_0 @ B_0u

    # the upper triangular matrix is restricted to sub-blocks
    T_1pp = T_1[:ens_dim, :ens_dim]
    T_1pm = T_1[:ens_dim, ens_dim:]
    T_1mm = T_1[ens_dim:, ens_dim:]

    # as is the model error matrix, assumed to be written in a basis of BLVs
    Q_ff = Q_1[:ens_dim, :ens_dim]
    Q_fu = Q_1[:ens_dim, ens_dim:]
    Q_uu = Q_1[ens_dim:, ens_dim:]

    # define the square root of the filtered variables for the update
    [U, S_0, V] = np.linalg.svd(S_0sqr)
    S_0 = U @ np.diag(np.sqrt(S_0)) @ U.T

    # precision matrix
    Omega0 = S_0 @ H_f.T @ np.linalg.inv(R) @ H_f @ S_0

    # reduced gain equation
    J_0 = S_0sqr @ H_f.T @ np.linalg.inv(H_f @ S_0sqr @ H_f.T + R)

    # UPDATE STEPS

    # filtered block update
    S_1sqr = T_1pp @ S_0 @ np.linalg.inv(I_f + Omega0) @ S_0 @ T_1pp.T + Q_ff

    # # we store the aus update version to compare the amount of inflation after the fact
    # S_AUS = copy.copy(S_1sqr)

    if unfil_d == 1:
        tmp = (T_1pm - np.reshape(T_1pp @ J_0 @ H_u, [ens_dim, 1]))
        sigma = P_0uu * tmp @ tmp.T
        Phi = T_1pp @ (I_f - J_0 @ H_f) @ P_0fu
        # reshape to get the exterior product of the two vectors and prevent unintended broadcasting
        Phi = np.reshape(Phi, [ens_dim, 1]) @ (T_1pm - np.reshape(T_1pp @ J_0 @ H_u, [ens_dim, 1])).T

        sigma += Phi + Phi.T
        S_1sqr += sigma
    else:
        sigma = (T_1pm - T_1pp @ J_0 @ H_u) @ P_0uu @ (T_1pm - T_1pp @ J_0 @ H_u).T
        Phi = T_1pp @ (I_f - J_0 @ H_f) @ P_0fu @ (T_1pm - T_1pp @ J_0 @ H_u).T
        sigma += Phi + Phi.T
        S_1sqr += sigma

    # unfiltered update step
    if unfil_d == 1:
        P_1uu = Q_uu + P_0uu * T_1mm**2
    else:
        P_1uu = Q_uu + T_1mm @ P_0uu @ T_1mm.T

    # cross covariance steps
    if unfil_d == 1:
        P_1fu = (T_1pm - np.reshape(T_1pp @ J_0 @ H_u, [ens_dim, 1])) * P_0uu * T_1mm + Q_fu
        P_1fu += np.reshape(T_1pp @ (I_f - J_0 @ H_f) @ P_0fu * T_1mm, [ens_dim, 1])
    else:
        P_1fu = (T_1pm - T_1pp @ J_0 @ H_u) @ P_0uu @ T_1mm.T + Q_fu
        P_1fu += T_1pp @ (I_f - J_0 @ H_f) @ P_0fu @ T_1mm.T

    # broadcast the updates into the matrix P_1
    P_1 = np.zeros([sys_dim, sys_dim])
    P_1[:ens_dim, :ens_dim] = S_1sqr * inflation
    P_1[:ens_dim, ens_dim:] = P_1fu
    P_1[ens_dim:, :ens_dim] = P_1fu.T
    P_1[ens_dim:, ens_dim:] = P_1uu

    return P_1

# ########################################################################################################################
# # simulate
#
#
# def simulate_ekf_ause_jump(x_0, truth, ens_dim, obs_dim, h, f, tanl, tl_fore_steps, model_err, obs_un, seed, burn):
#     """This function simulates the extended kalman filter with alternating obs operator"""
#
#     # define initial parameters
#     [sys_dim, nanl] = np.shape(truth)
#     lle = np.zeros(ens_dim)
#
#     # define the observations
#     H = alt_obs(sys_dim, obs_dim)
#     obs_seed = seed + 10000
#     obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
#     P_0 = np.eye(ens_dim) * model_err
#     R = np.eye(obs_dim) * obs_un
#     Q = np.eye(ens_dim) * model_err * tanl**2
#     B_0 = np.eye(sys_dim, M=ens_dim)
#
#     x_fore = np.zeros([sys_dim, nanl])
#     x_anal = np.zeros([sys_dim, nanl])
#
#     for i in range(nanl):
#         # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
#         # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model
#
#         # initialize the tangent linear model
#         B = B_0
#         x = x_0
#
#         for j in range(tl_fore_steps):
#             [x, B] = l96_step_TLM(x, B, h, f, l96_rk4_step)
#
#         # define the forecast state
#         x_1 = x
#         x_fore[:, i] = x_1
#
#         # compute the backward Lyapunov vectors at the next step recursively
#         B_1, T_1 = np.linalg.qr(B)
#         lle += np.log(np.abs(np.diagonal(T_1)))
#
#         # the AUSE Riccati equation is defined in terms of the above values
#         P_1 = EKFAUSE_fore(P_0, B_0, T_1, H, Q, R)
#
#         # update the ensemble span of the leading BLVs and the associated observation operator for the analysis time
#         B_f = B_1[:, :-1]
#         H_f = H.dot(B_f)
#
#         # define the kalman gain and find the analysis mean with the new forecast uncertainty
#         S_sqr = P_1[:-1, :-1]
#         K = B_f @ S_sqr @ H_f.T @ np.linalg.inv(H_f @ S_sqr @ H_f.T + R)
#
#         x_1 = x_1 + K @ (obs[:, i] - H @ x_1)
#         x_anal[:, i] = x_1
#
#         # re-initialize
#         x_0 = x_1
#         P_0 = P_1
#         B_0 = B_1
#
#     # compute the rmse for the forecast and analysis
#     fore = x_fore - truth
#     fore = np.sqrt(np.mean(fore * fore, axis=0))
#
#     anal = x_anal - truth
#     anal = np.sqrt(np.mean(anal * anal, axis=0))
#
#     f_rmse = []
#     a_rmse = []
#
#     for i in range(burn + 1, nanl):
#         f_rmse.append(np.mean(fore[burn: i]))
#         a_rmse.append(np.mean(anal[burn: i]))
#
#     le = lle/(nanl * tl_fore_steps * h)
#
#     return {'fore': f_rmse, 'anal': a_rmse, 'le': le}
#
# ########################################################################################################################


def simulate_ekf_aus_jump(x_0, truth, ens_dim, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un, seed, burn, m_inf=1.0,
                          a_inf=0.0, clim=[0]):
    """This function simulates the extended kalman AUS filter"""

    # define initial parameters
    [sys_dim, nanl] = np.shape(truth)

    # climatological variance place holder, to be compatible with or without additive inflaiton
    if len(clim) == 1:
        clim = np.zeros([sys_dim, sys_dim])

    lle = np.zeros(ens_dim)

    # define the observations
    H = alt_obs(sys_dim, obs_dim)
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
    R = np.eye(obs_dim) * obs_un
    P_0 = np.eye(ens_dim)
    Q = Q * tanl**2
    B_0 = np.eye(sys_dim, M=ens_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
        # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model

        # initialize the perturbations in the tl model
        B = B_0
        x = x_0
        for j in range(tl_fore_steps):
            [x, B] = l96_step_TLM(x, B, h, f, l96_rk4_step)

        # define the forecast state
        x_1 = x
        x_fore[:, i] = x_1

        # compute the backward Lyapunov vectors at the next step recursively
        B_1, T_1 = np.linalg.qr(B)
        lle += np.log(np.abs(np.diagonal(T_1)))

        # we write Q in the basis of BLV
        Q_tmp = B_1.T @ Q @ B_1

        # the AUS Riccati equation is defined in terms of the above values
        P_1 = EKFAUS_fore(P_0, B_0, T_1, H, Q_tmp, R, inflation=m_inf)

        # we define the orthogonal-complement basis to the ensemble, and use it to define the additive inflation
        o_comp = np.eye(sys_dim) - B_1.dot(B_1.transpose())
        [eigs, C] = np.linalg.eigh(o_comp)
        C = C[:, ens_dim:]
        add_inf = a_inf * C.T @ clim @ C

        # update the ensemble span of the leading BLVs and the associated observation operator for the analysis time,
        # likewise the orthogonal complement and the respective operator
        H_b = H.dot(B_1)
        H_c = H.dot(C)

        # define the kalman gain and find the analysis mean with the new forecast uncertainty
        K = np.linalg.inv(H_b @ P_1 @ H_b.T + H_c @ add_inf @ H_c.T + R)
        K = B_1 @ P_1 @ H_b.T @ K + C @ add_inf @ H_c.T @ K
        x_1 = x_1 + K @ (obs[:, i] - H @ x_1)
        x_anal[:, i] = x_1

        # re-initialize
        x_0 = x_1
        P_0 = P_1
        B_0 = B_1

    # compute the rmse for the forecast and analysis
    fore = x_fore - truth
    fore = np.sqrt(np.mean(fore * fore, axis=0))

    anal = x_anal - truth
    anal = np.sqrt(np.mean(anal * anal, axis=0))

    f_rmse = []
    a_rmse = []

    for i in range(burn + 1, nanl):
        f_rmse.append(np.mean(fore[burn: i]))
        a_rmse.append(np.mean(anal[burn: i]))

    le = lle/(nanl * tl_fore_steps * h)

    return {'fore': f_rmse, 'anal': a_rmse, 'le': le}

########################################################################################################################


def simulate_ekf_aus_sde(x_0, truth, ens_dim, obs_dim, h, f, tl_fore_steps, diffusion, obs_un, seed, burn, m_inf=1.0,
                          a_inf=0.0, clim=[0]):
    """This function simulates the extended kalman AUS filter with alternating obs operator"""

    # define initial parameters
    [sys_dim, nanl] = np.shape(truth)

    # climatological variance place holder, to be compatible with or without additive inflaiton
    if len(clim) == 1:
        clim = np.zeros([sys_dim, sys_dim])

    lle = np.zeros(ens_dim)

    # define the observations
    H = alt_obs(sys_dim, obs_dim)
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
    R = np.eye(obs_dim) * obs_un
    P_0 = np.eye(ens_dim)
    B_0 = np.eye(sys_dim, M=ens_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
        # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model

        # initialize the perturbations in the tl model
        B = B_0
        x = x_0
        B_hist = np.zeros([sys_dim, ens_dim, tl_fore_steps])

        for j in range(tl_fore_steps):
            # we integrate with respect to the 2 order taylor in the nonlinear
            [x, B] = l96_step_TLM(x, B, h, f, l96_2tay_step)
            B_hist[:, :, j] = B

        # define the forecast state
        x_1 = x
        x_fore[:, i] = x_1

        # compute the backward Lyapunov vectors at the next step recursively
        B_1, T_1 = np.linalg.qr(B)
        lle += np.log(np.abs(np.diagonal(T_1)))

        # Q is defined by the forward linear evolved wiener process covariance scaled by diffusion squared
        # (we are assuming that diffusion is modelled by a scalar matrix, which is applied to the standard normal
        # weiner process).  We estimate the integral with the left Reimann sum

        # left most value
        Q = T_1 @ T_1.T * h

        for j in range(tl_fore_steps - 1):
            # sum through the remaining values in the interval
            B_tmp, T_tmp = np.linalg.qr(np.squeeze(B_hist[:, :, j]))
            PHI = T_1 @ np.linalg.inv(T_tmp)
            Q += PHI @ PHI.T * h

        Q = diffusion ** 2 * Q

        # the AUS Riccati equation is defined in terms of the above values
        P_1 = EKFAUS_fore(P_0, B_0, T_1, H, Q, R, inflation=m_inf)

        # we define the orthogonal-complement basis to the ensemble, and use it to define the additive inflation
        o_comp = np.eye(sys_dim) - B_1.dot(B_1.transpose())
        [eigs, C] = np.linalg.eigh(o_comp)
        C = C[:, ens_dim:]
        add_inf = a_inf * C.T @ clim @ C

        # update the ensemble span of the leading BLVs and the associated observation operator for the analysis time,
        # likewise the orthogonal complement and the respective operator
        H_b = H.dot(B_1)
        H_c = H.dot(C)

        # define the kalman gain and find the analysis mean with the new forecast uncertainty
        K = np.linalg.inv(H_b @ P_1 @ H_b.T + H_c @ add_inf @ H_c.T + R)
        K = B_1 @ P_1 @ H_b.T @ K + C @ add_inf @ H_c.T @ K
        x_1 = x_1 + K @ (obs[:, i] - H @ x_1)
        x_anal[:, i] = x_1

        # re-initialize
        x_0 = x_1
        P_0 = P_1
        B_0 = B_1

    # compute the rmse for the forecast and analysis
    fore = x_fore - truth
    fore = np.sqrt(np.mean(fore * fore, axis=0))

    anal = x_anal - truth
    anal = np.sqrt(np.mean(anal * anal, axis=0))

    f_rmse = []
    a_rmse = []

    for i in range(burn + 1, nanl):
        f_rmse.append(np.mean(fore[burn: i]))
        a_rmse.append(np.mean(anal[burn: i]))

    le = lle/(nanl * tl_fore_steps * h)

    return {'fore': f_rmse, 'anal': a_rmse, 'le': le}

########################################################################################################################

def simulate_kf_ause_jump(x_0, truth, ens_dim, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un, seed, burn, infl=1.0):
    """This function simulates the extended kalman filter with alternating obs operator"""

    # define initial parameters
    [sys_dim, nanl] = np.shape(truth)
    lle = np.zeros(sys_dim)

    # define the observations
    H = alt_obs(sys_dim, obs_dim)
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
    P_0 = np.eye(sys_dim)
    R = np.eye(obs_dim) * obs_un
    Q = Q * tanl**2
    B_0 = np.eye(sys_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
        # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model

        #initialize the tangent linear model
        x = x_0
        B = B_0

        for j in range(tl_fore_steps):
            [x, B] = l96_step_TLM(x, B, h, f, l96_rk4_step)

        # define the forecast state
        x_1 = x
        x_fore[:, i] = x_1

        # compute the backward Lyapunov vectors at the next step recursively
        B_1, T_1 = np.linalg.qr(B)
        lle += np.log(np.abs(np.diagonal(T_1)))

        # we write Q in the basis of BLV
        Q_tmp = B_1.T @ Q @ B_1

        # the AUSE Riccati equation is defined in terms of the above values
        P_1 = KFAUSE_fore(P_0, B_0, T_1, H, ens_dim, Q_tmp, R, inflation=infl)

        # # find the eigenvalues of the inflation transformations
        # addi_inf[:, i] = np.linalg.eigvalsh(sigma)
        # mult_inf[:, i] = np.linalg.eigvals(P_1[:ens_dim, :ens_dim] @ np.linalg.inv(S_AUS))
        # sigma_diagonal[:, i] = np.diagonal(sigma)

        # update the ensemble span of the leading BLVs and the associated observation operator for the analysis time
        B_f = B_1[:, :ens_dim]
        H_f = H.dot(B_f)

        # define the kalman gain and find the analysis mean with the new forecast uncertainty
        S_sqr = P_1[:ens_dim, :ens_dim]
        K = B_f @ S_sqr @ H_f.T @ np.linalg.inv(H_f @ S_sqr @ H_f.T + R)

        x_1 = x_1 + K @ (obs[:, i] - H @ x_1)
        x_anal[:, i] = x_1

        # re-initialize
        x_0 = x_1
        P_0 = P_1
        B_0 = B_1

    # compute the rmse for the forecast and analysis
    fore = x_fore - truth
    fore = np.sqrt(np.mean(fore * fore, axis=0))

    anal = x_anal - truth
    anal = np.sqrt(np.mean(anal * anal, axis=0))

    f_rmse = []
    a_rmse = []

    for i in range(burn + 1, nanl):
        f_rmse.append(np.mean(fore[burn: i]))
        a_rmse.append(np.mean(anal[burn: i]))

    le = lle/(nanl * tl_fore_steps * h)

    return {'fore': f_rmse, 'anal': a_rmse, 'le': le}

########################################################################################################################


def simulate_kf_ause_sde(x_0, truth, ens_dim, obs_dim, h, f, tl_fore_steps, diffusion, obs_un, seed, burn, infl=1.0):
    """This function simulates the extended kalman filter with alternating obs operator"""

    # define initial parameters
    [sys_dim, nanl] = np.shape(truth)
    lle = np.zeros(sys_dim)

    # define the observations
    H = alt_obs(sys_dim, obs_dim)
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
    P_0 = np.eye(sys_dim)
    R = np.eye(obs_dim) * obs_un
    B_0 = np.eye(sys_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
        # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model

        #initialize the tangent linear model
        x = x_0
        B = B_0
        B_hist = np.zeros([sys_dim, sys_dim, tl_fore_steps])

        for j in range(tl_fore_steps):
            [x, B] = l96_step_TLM(x, B, h, f, l96_2tay_step)
            B_hist[:, :, j] = B

        # define the forecast state
        x_1 = x
        x_fore[:, i] = x_1

        # compute the backward Lyapunov vectors at the next step recursively
        B_1, T_1 = np.linalg.qr(B)
        lle += np.log(np.abs(np.diagonal(T_1)))

        # Q is defined by the forward linear evolved wiener process covariance scaled by diffusion squared
        # (we are assuming that diffusion is modelled by a scalar matrix, which is applied to the standard normal
        # weiner process).  We estimate the integral with the left Reimann sum
        Q = T_1 @ T_1.T * h

        for j in range(tl_fore_steps - 1):
            B_tmp, T_tmp = np.linalg.qr(np.squeeze(B_hist[:, :, j]))
            PHI = T_1 @ np.linalg.inv(T_tmp)
            Q += PHI @ PHI.T * h

        Q = diffusion ** 2 * Q
        # the AUSE Riccati equation is defined in terms of the above values
        P_1 = KFAUSE_fore(P_0, B_0, T_1, H, ens_dim, Q, R, inflation=infl)

        # # find the eigenvalues of the inflation transformations
        # addi_inf[:, i] = np.linalg.eigvalsh(sigma)
        # mult_inf[:, i] = np.linalg.eigvals(P_1[:ens_dim, :ens_dim] @ np.linalg.inv(S_AUS))
        # sigma_diagonal[:, i] = np.diagonal(sigma)

        # update the ensemble span of the leading BLVs and the associated observation operator for the analysis time
        B_f = B_1[:, :ens_dim]
        H_f = H.dot(B_f)

        # define the kalman gain and find the analysis mean with the new forecast uncertainty
        S_sqr = P_1[:ens_dim, :ens_dim]
        K = B_f @ S_sqr @ H_f.T @ np.linalg.inv(H_f @ S_sqr @ H_f.T + R)

        x_1 = x_1 + K @ (obs[:, i] - H @ x_1)
        x_anal[:, i] = x_1

        # re-initialize
        x_0 = x_1
        P_0 = P_1
        B_0 = B_1

    # compute the rmse for the forecast and analysis
    fore = x_fore - truth
    fore = np.sqrt(np.mean(fore * fore, axis=0))

    anal = x_anal - truth
    anal = np.sqrt(np.mean(anal * anal, axis=0))

    f_rmse = []
    a_rmse = []

    for i in range(burn + 1, nanl):
        f_rmse.append(np.mean(fore[burn: i]))
        a_rmse.append(np.mean(anal[burn: i]))

    le = lle/(nanl * tl_fore_steps * h)

    return {'fore': f_rmse, 'anal': a_rmse, 'le': le}
# ########################################################################################################################
# # simulate
#
#
# def simulate_ekf_ause_alt_jump(x_0, truth, ens_dim, obs_dim, h, f, tanl, tl_fore_steps, model_err, obs_un, seed, burn):
#     """This function simulates the extended kalman filter with alternating obs operator"""
#
#     # define initial parameters
#     [sys_dim, nanl] = np.shape(truth)
#     lle = np.zeros(ens_dim)
#
#     # define the observations
#     H = alt_obs(sys_dim, obs_dim)
#     obs_seed = seed + 10000
#     obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
#     P_0 = np.eye(ens_dim) * model_err
#     R = np.eye(obs_dim) * obs_un
#     Q = np.eye(ens_dim) * model_err * tanl**2
#     B_0 = np.eye(sys_dim, M=ens_dim)
#
#     x_fore = np.zeros([sys_dim, nanl])
#     x_anal = np.zeros([sys_dim, nanl])
#
#     for i in range(nanl):
#         # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
#         # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model
#
#         # initialize the tangent linear model
#         B = B_0
#         x = x_0
#
#         for j in range(tl_fore_steps):
#             [x, B] = l96_step_TLM(x, B, h, f, l96_rk4_step)
#
#         # define the forecast state
#         x_1 = x
#         x_fore[:, i] = x_1
#
#         # compute the backward Lyapunov vectors at the next step recursively
#         B_1, T_1 = np.linalg.qr(B)
#         lle += np.log(np.abs(np.diagonal(T_1)))
#
#         # the AUSE Riccati equation is defined in terms of the above values
#         P_1 = EKFAUSE_fore(P_0, B_0, T_1, H, Q, R)
#
#         # HERE IN THE ALTERNATE VERSION, WE MAKE A FULL RANK CORRECTION
#
#         # # update the ensemble span of the leading BLVs and the associated observation operator for the analysis time
#         # B_f = B_1[:, :-1]
#         # H_f = H.dot(B_f)
#         #
#         # # define the kalman gain and find the analysis mean with the new forecast uncertainty
#         # S_sqr = P_1[:-1, :-1]
#         # K = B_f @ S_sqr @ H_f.T @ np.linalg.inv(H_f @ S_sqr @ H_f.T + R)
#
#         H_f = H.dot(B_1)
#         K = B_1 @ P_1 @ H_f.T @ np.linalg.inv(H_f @ P_1 @ H_f.T + R)
#
#         x_1 = x_1 + K @ (obs[:, i] - H @ x_1)
#         x_anal[:, i] = x_1
#
#         # re-initialize
#         x_0 = x_1
#         P_0 = P_1
#         B_0 = B_1
#
#     # compute the rmse for the forecast and analysis
#     fore = x_fore - truth
#     fore = np.sqrt(np.mean(fore * fore, axis=0))
#
#     anal = x_anal - truth
#     anal = np.sqrt(np.mean(anal * anal, axis=0))
#
#     f_rmse = []
#     a_rmse = []
#
#     for i in range(burn + 1, nanl):
#         f_rmse.append(np.mean(fore[burn: i]))
#         a_rmse.append(np.mean(anal[burn: i]))
#
#     le = lle/(nanl * tl_fore_steps * h)
#
#     return {'fore': f_rmse, 'anal': a_rmse, 'le': le}
#
# ########################################################################################################################


