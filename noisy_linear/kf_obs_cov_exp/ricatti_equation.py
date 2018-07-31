import numpy as np


def ricatti_evo(M_kl, P, model_err, obs_err, H_transpose, burn=0):
    """"This function returns a sequence of analysis error covariance matrices for the ricatti equation"""

    [sys_dim, obs_dim, nanl] = np.shape(H_transpose)
    P_eigs = np.zeros([sys_dim, nanl - burn])
    Q = model_err * np.eye(sys_dim)
    R = obs_err * np.eye(obs_dim)
    I = np.eye(sys_dim)

    for i in range(nanl):
        # define the current operators for the ricatti equation
        H_0 = np.squeeze(H_transpose[:, :, i]).T
        M_1 = np.squeeze(M_kl[:, :, i])

        # compute the forecast
        P = kf_fore(M_1, P, Q, R, H_0, I)
        if i >= burn:
            # store eigenvalues after burn-in for the DA routine
            P_eigs[:, i - burn] = sorted(np.linalg.eigvalsh(P))[::-1]

    return P_eigs


########################################################################################################################
# forecast ricatti equation


def kf_fore(M_1, P_0, Q_1, R_0, H_0, I):
    """"This function returns a sequence of analysis error covariance matrices for the riccati equation"""

    # the forecast riccati is given in the symmetric square root form
    [U_0, S_0, V] = np.linalg.svd(P_0)
    X_0 = U_0 @ np.diag(np.sqrt(S_0)) @ U_0.T

    Omega = H_0.T @ np.linalg.inv(R_0) @ H_0
    XOX = X_0.T @ Omega @ X_0

    P_1 = M_1 @ X_0 @ np.linalg.inv(I + XOX) @ X_0.T @ M_1.T + Q_1

    return P_1
