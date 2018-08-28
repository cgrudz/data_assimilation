import numpy as np
from scipy.linalg import schur
from observation_operators import random_obs
from observation_operators import id_obs
from observation_operators import dual_subspace_obs
from observation_operators import pseudo_inv_obs


########################################################################################################################
# forecast ricatti equation


def kf_fore(M_kl, P_init, model_err, obs_err, H_transpose):
    """"This function returns a sequence of analysis error covariance matrices for the ricatti equation"""

    [sys_dim, sys_dim, nfor] = np.shape(M_kl)
    [sys_dim, obs_dim, nobs] = np.shape(H_transpose)
    P_traj = np.zeros([sys_dim, sys_dim, nfor + 1])

    P_traj[:, :, 0] = P_init
    Q = model_err * np.eye(sys_dim)
    R = obs_err * np.eye(obs_dim)
    I = np.eye(sys_dim)

    for i in range(nfor):
        # define the current operators for the Riccati equation
        H_0 = np.squeeze(H_transpose[:, :, i]).T
        M_1 = np.squeeze(M_kl[:, :, i])
        P_0 = np.squeeze(P_traj[:, :, i])

        [U, s, V] = np.linalg.svd(P_0)
        X_0 = U @ np.diag(np.sqrt(s))

        Omega_0 = H_0.T @ np.linalg.inv(R) @ H_0
        XOX = X_0.T @ Omega_0 @ X_0

        P_1 = M_1 @ X_0 @ np.linalg.inv(I + XOX) @ X_0.T @ M_1.T + Q
        P_traj[:, :, i+1] = P_1

    return P_traj

########################################################################################################################
# forecast analysis function


def analyze_p(P, BLVs, FLVs, CLVs):
    """" This is a quick script to find the eigenvalues and the projections into the Laypunov vectos"""

    [sys_dim, foo, fore_num] = np.shape(P)
    nanl = fore_num - 1

    p_eigs = []
    p_bb = []
    p_ff = []
    p_cc = []

    # we find all eigenvalues and find the inner product with the BLVS, FLVs and CLVS
    for i in range(nanl):
        tmp_p = np.linalg.eigvalsh(np.squeeze(P[:, :, i + 1]))
        p_eigs.append(tmp_p[::-1])
        tmp_b = []
        tmp_f = []
        tmp_c = []

        for j in range(sys_dim):
            b_j = np.squeeze(BLVs[:, j, i])
            f_j = np.squeeze(FLVs[:, j, i])
            c_j = np.squeeze(CLVs[:, j, i])

            tmp_b.append(b_j.T @ np.squeeze(P[:, :, i + 1]) @ b_j)
            tmp_f.append(f_j.T @ np.squeeze(P[:, :, i + 1]) @ f_j)
            tmp_c.append(c_j.T @ np.squeeze(P[:, :, i + 1]) @ c_j)

        p_bb.append(tmp_b)
        p_ff.append(tmp_f)
        p_cc.append(tmp_c)

    return [np.array(p_eigs), np.array(p_bb), np.array(p_ff), np.array(p_cc)]


########################################################################################################################
# simulate


def simulate_kf(FLVs, BLVs, CLVs, sys_dim, obs_dim, M_kl, model_err, obs_un, nanl, seed):
    """This function simulates the observational models using FLV,BLV,ID and rand obs"""

    # NOTE: WE ALWAYS ASSUME THE INITIAL UNCERTAINTY TO BE THE MODEL ERROR
    P_init = np.eye(sys_dim) * model_err

    # ####################################################################################################################
    # # forward
    # H = FLVs[:, :obs_dim, :]
    # P = kf_fore(M_kl, P_init, model_err, obs_un, H)
    # p_f = analyze_p(P, BLVs, FLVs, CLVs)
    #
    # del P
    # del H
    # ####################################################################################################################
    # # backward
    # H = BLVs[:, :obs_dim, :]
    # P = kf_fore(M_kl, P_init, model_err, obs_un, H)
    # p_b = analyze_p(P, BLVs, FLVs, CLVs)
    #
    # del P
    # del H
    #
    # ####################################################################################################################
    # # arbitrary indices
    #
    # H = id_obs(sys_dim, obs_dim, nanl)
    # P = kf_fore(M_kl, P_init, model_err, obs_un, H)
    # p_I = analyze_p(P, BLVs, FLVs, CLVs)
    #
    # del P
    # del H

    ####################################################################################################################
    # random
    H = random_obs(sys_dim, obs_dim, nanl, seed)
    P = kf_fore(M_kl, P_init, model_err, obs_un, H)
    p_r = analyze_p(P, BLVs, FLVs, CLVs)

    del P
    del H

    # ####################################################################################################################
    # # covariant normalized
    # H, C_s_vals, C_dual_s_vals = dual_subspace_obs(CLVs, obs_dim)
    # P = kf_fore(M_kl, P_init, model_err, obs_un, H)
    # p_c = analyze_p(P, BLVs, FLVs, CLVs)
    # p_c.append([C_s_vals, C_dual_s_vals])
    #
    # del P
    # del H

    # ####################################################################################################################
    # # covariant un-normalized
    # H = dual_subspace_obs(CLVs, obs_dim, normalized=False)
    # P = kf_fore(M_kl, P_init, model_err, obs_un, H)
    # p_c_un = analyze_p(P, BLVs, FLVs, CLVs)
    #
    # del P
    # del H
    #
    # ####################################################################################################################
    # # covariant pseudo inverse normalized
    # H = pseudo_inv_obs(CLVs[:, :obs_dim, :])
    # P = kf_fore(M_kl, P_init, model_err, obs_un, H)
    # p_cp = analyze_p(P, BLVs, FLVs, CLVs)
    #
    # del P
    # del H
    #
    # ####################################################################################################################
    # # covariant pseudo inverse un-normalized
    # H = pseudo_inv_obs(CLVs[:, :obs_dim, :], normalized=False)
    # P = kf_fore(M_kl, P_init, model_err, obs_un, H)
    # p_cp_un = analyze_p(P, BLVs, FLVs, CLVs)
    #
    # del P
    # del H
    #
    # ####################################################################################################################
    # return a dictionary of results

    return {'b': p_b, 'f': p_f, 'r': p_r, 'i': p_I, 'c': p_c} #, 'c_un': p_c_un, 'cp': p_cp, 'cp_un': p_cp_un}
