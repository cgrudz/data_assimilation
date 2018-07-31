import numpy as np

########################################################################################################################
# analysis ricatti equation


def ricatti_anal_evo(M_kl, P_init, model_err, obs_err, H_transpose):
    """"This function returns a sequence of analysis error covariance matrices for the ricatti equation"""

    [sys_dim, obs_dim, nanl] = np.shape(H_transpose)
    P_traj = np.zeros([sys_dim, sys_dim, nanl+1])
    P_traj[:, :, 0] = P_init
    Q = model_err * np.eye(sys_dim)
    R = obs_err * np.eye(obs_dim)

    for i in range(nanl):
        # define the current operators for the ricatti equation
        H = np.squeeze(H_transpose[:, :, i]).T
        M = np.squeeze(M_kl[:, :, i])
        P = np.squeeze(P_traj[:, :, i])

        P = M.dot(P)
        P = P.dot(M.T) + Q

        KG = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        KH = KG @ H
        P = (np.eye(sys_dim) - KH) @ P

        P_traj[:, :, i+1] = P

    return [P_traj]

########################################################################################################################
# forecast ricatti equation


def ricatti_fore_evo(M_kl, P_init, model_err, obs_err, H_transpose):
    """"This function returns a sequence of analysis error covariance matrices for the ricatti equation"""

    [sys_dim, sys_dim, nfor] = np.shape(M_kl)
    [sys_dim, obs_dim, nobs] = np.shape(H_transpose)
    P_traj = np.zeros([sys_dim, sys_dim, nfor + 1])

    P_traj[:, :, 0] = P_init
    Q = model_err * np.eye(sys_dim)
    R = obs_err * np.eye(obs_dim)
    I = np.eye(sys_dim)
    alpha = np.zeros([nfor])
    beta = np.zeros([nfor])

    for i in range(nfor):
        # define the current operators for the Riccati equation
        H_0 = np.squeeze(H_transpose[:, :, i]).T
        M_1 = np.squeeze(M_kl[:, :, i])
        P_0 = np.squeeze(P_traj[:, :, i])

        [U, s, V] = np.linalg.svd(P_0)
        X_0 = U @ np.diag(np.sqrt(s))

        Omega_0 = H_0.T @ np.linalg.inv(R) @ H_0
        XOX = X_0.T @ Omega_0 @ X_0
        eigs = np.linalg.eigvalsh(XOX)
        alpha[i] = min(eigs)
        beta[i] = max(eigs)

        P_1 = M_1 @ X_0 @ np.linalg.inv(I + XOX) @ X_0.T @ M_1.T + Q
        P_traj[:, :, i+1] = P_1

    return [P_traj, alpha, beta]

########################################################################################################################
# forecast sqrt ricatti equation


def ricatti_fore_sqrt_evo(T_k, S_sqr_init, model_err, obs_err, nanl):
    """"This function returns a sequence of analysis error covariance matrices for the sqrt ricatti equation

    Note: we make the reduction in the calculation that R, Q are diagonal and time invariant.  The T matrix can be
    full dimensional, it will be reduced to the ensemble dimension."""

    [ens_dim, foo] = np.shape(S_sqr_init)
    S_sqr_traj = np.zeros([ens_dim, ens_dim, nanl+1])

    S_sqr_traj[:, :, 0] = S_sqr_init
    Q = model_err * np.eye(ens_dim)
    R = obs_err * np.eye(ens_dim)
    I = np.eye(ens_dim)
    alpha = np.zeros([nanl])
    beta = np.zeros([nanl])

    for i in range(nanl):
        # define the current operators for the ricatti equation
        T = np.squeeze(T_k[:ens_dim, :ens_dim, i])
        S_sqr = np.squeeze(S_sqr_traj[:, :, i])

        [U, S, V] = np.linalg.svd(S_sqr)
        S = U @ np.diag(np.sqrt(S)) @ U.T

        Omega = S @ np.linalg.inv(R) @ S
        eigs = np.linalg.eigvalsh(Omega)
        alpha[i] = min(eigs)
        beta[i] = max(eigs)

        S_sqr = T @ S @ np.linalg.inv(I + Omega) @ S @ T.T + Q
        S_sqr_traj[:, :, i+1] = S_sqr

    return [S_sqr_traj, alpha, beta]

########################################################################################################################

# random observational operator


def random_obs(sys_dim, obs_dim, nanl, first_seed):

    H = np.zeros([sys_dim, obs_dim, nanl])

    for i in range(nanl):
        np.random.seed(first_seed + 1)
        tmp = np.random.randn(sys_dim, obs_dim)
        tmp = np.linalg.qr(tmp)
        H[:, :, i] = tmp[0]

    return H

########################################################################################################################
# identity slice operator


def id_obs(sys_dim, obs_dim, nanl):
    H_I = np.zeros([sys_dim, obs_dim, nanl])
    for i in range(nanl):
        H_I[:, :, i] = np.eye(sys_dim, M=obs_dim)

    return H_I

########################################################################################################################
# simulate


def simulate(FLVs, BLVs, sys_dim, obs_dim, M_kl, P_init, model_err, obs_un, nanl, seed, ricatti, forecast=False):
    """This function simulates the observational models using FLV,BLV,ID and rand obs"""

    ####################################################################################################################
    # forward
    H_f = FLVs[:, :obs_dim, :]
    tmp = ricatti(M_kl, P_init, model_err, obs_un, H_f)
    P_f = tmp[0]
    p_f = []
    bp = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        tmp_p = np.linalg.eigvalsh(np.squeeze(P_f[:, :, i + 1]))
        p_f.append(tmp_p)

        # find the inner product of the kth blv through the covariance at time i+1
        for k in range(sys_dim):
            b = np.squeeze(BLVs[:, k, i + 1])
            bp[k, i] = b.T @ np.squeeze(P_f[:, :, i + 1]) @ b

    p_f = {'p': np.array(p_f), 'bp': bp}

    if forecast:
        p_f['alpha'] = tmp[1]
        p_f['beta'] = tmp[2]

    del P_f
    del H_f
    del tmp

    ####################################################################################################################
    # backward
    H_b = BLVs[:, :obs_dim, :]
    tmp = ricatti(M_kl, P_init, model_err, obs_un, H_b)
    P_b = tmp[0]
    p_b = []
    bp = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        tmp_p = np.linalg.eigvalsh(np.squeeze(P_b[:, :, i + 1]))
        p_b.append(tmp_p)

        # find the inner product of the kth blv through the covariance at time i+1
        for k in range(sys_dim):
            b = np.squeeze(BLVs[:, k, i + 1])
            bp[k, i] = b.T @ np.squeeze(P_b[:, :, i + 1]) @ b

    p_b = {'p': np.array(p_b), 'bp': bp}

    if forecast:
        p_b['alpha'] = tmp[1]
        p_b['beta'] = tmp[2]

    del P_b
    del H_b
    del tmp

    ####################################################################################################################
    # arbitrary indices

    H_I = id_obs(sys_dim, obs_dim, nanl)
    tmp = ricatti(M_kl, P_init, model_err, obs_un, H_I)
    P_I = tmp[0]
    p_I = []
    bp = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        tmp_p = np.linalg.eigvalsh(np.squeeze(P_I[:, :, i + 1]))
        p_I.append(tmp_p)

        # find the inner product of the kth blv through the covariance at time i+1
        for k in range(sys_dim):
            b = np.squeeze(BLVs[:, k, i + 1])
            bp[k, i] = b.T @ np.squeeze(P_I[:, :, i + 1]) @ b

    p_I = {'p': np.array(p_I), 'bp': bp}

    if forecast:
        p_I['alpha'] = tmp[1]
        p_I['beta'] = tmp[2]

    del P_I
    del H_I
    del tmp

    ####################################################################################################################
    # random, two components
    H_r = random_obs(sys_dim, obs_dim, nanl, seed)

    tmp = ricatti(M_kl, P_init, model_err, obs_un, H_r)
    P_r = tmp[0]
    p_r = []
    bp = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        tmp_p = np.linalg.eigvalsh(np.squeeze(P_r[:, :, i + 1]))
        p_r.append(tmp_p)

        # find the inner product of the kth blv through the covariance at time i+1
        for k in range(sys_dim):
            b = np.squeeze(BLVs[:, k, i + 1])
            bp[k, i] = b.T @ np.squeeze(P_r[:, :, i + 1]) @ b

    p_r = {'p': np.array(p_r), 'bp': bp}

    if forecast:
        p_r['alpha'] = tmp[1]
        p_r['beta'] = tmp[2]

    del P_r
    del H_r
    del tmp

    return {'b': p_b, 'f': p_f, 'r': p_r, 'i': p_I}
