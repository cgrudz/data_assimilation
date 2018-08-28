import numpy as np
from scipy.linalg import schur

########################################################################################################################
# unfiltered blvs


def unfil_blv(T_minus):
    """This function computes the unfiltered covariance in the trailing blvs.

    We compute the running sum of the stable block symmetric products, looping until the first summand reaches numerical
    zero."""

    [s_dim, foo, nanl] = np.shape(T_minus)

    # define the storage for the unfiltered error covariance
    unfil_traj = np.zeros([s_dim, s_dim, nanl + 1])

    for k in range(nanl + 1):
        T_kl = np.eye(s_dim)
        unfil = np.eye(s_dim)

        for l in range(k-1, -1, -1):
            # we compute the product T_minus_k-1 @ ... @ T_minus_k-l recursively for each time l = 1, ... , k
            T_kl = np.dot(T_kl, np.squeeze(T_minus[:, :, l]))
            tmp_prod = np.dot(T_kl, T_kl.T)

            # if the computation becomes trivial stop
            if np.all(tmp_prod) <= 0:
                break

            # otherwise we add this to the sum of the uncertainties
            unfil = unfil + tmp_prod

        # store the unfiltered covariance at time k
        unfil_traj[:, :, k] = unfil[:]

    return unfil_traj

########################################################################################################################
# forecast analysis function


def analyze_p(P):
    """" This is a quick script to find the eigenvalues and the projections into the Lyapunov vectos"""

    [sys_dim, foo, fore_num] = np.shape(P)
    p_eigs = []
    p_bb = []

    # we find all eigenvalues and find the inner product with the BLVS
    for i in range(1, fore_num):
        tmp_p = np.linalg.eigvalsh(np.squeeze(P[:, :, i]))
        p_eigs.append(tmp_p[::-1])
        tmp_b = []

        for j in range(sys_dim):
            tmp_b.append(np.squeeze(P[j, j, i]))

        p_bb.append(tmp_b)

    return [np.array(p_eigs), np.array(p_bb)]


########################################################################################################################
# augmented KFAUSE, full forecast error Riccati equation


def KFAUSE_fore(B, T, H, ens_dim, model_err, obs_err):
    """"This function returns a sequence of error covariance matrices for the forecast error increment of KF-AUSE

    Note: we make the reduction in the calculation that R, Q are diagonal and time invariant.  T_k will be the full
    triangular matrix and BLVs will be the full basis of BLVs."""

    # infer parameters define storage and parameters for the experiment

    # H is given by the transpose of a slice by convention
    [sys_dim, obs_dim, nfore] = np.shape(H)
    unfil_d = sys_dim - ens_dim
    R = obs_err * np.eye(obs_dim)
    I_f = np.eye(ens_dim)
    I_u = np.eye(unfil_d)

    # P_traj will give the sequence of solutions to the Riccati equation, initialized with the model error covariance
    P_traj = np.zeros([sys_dim, sys_dim, nfore+1])
    P_traj[:, :, 0] = np.eye(sys_dim) * model_err

    for i in range(nfore):
        # define the current operators for the ricatti equation update

        # we separate the filtered and unfiltered parts of the last covariance
        P_0 = np.squeeze(P_traj[:, :, i])
        v = np.linalg.eigvalsh(P_0)
        if min(v) < 1:
            print(v)
        P_0uu = np.squeeze(P_0[ens_dim:, ens_dim:])
        P_0fu = np.squeeze(P_0[:ens_dim, ens_dim:])
        S_0sqr = np.squeeze(P_0[:ens_dim, :ens_dim])

        # define the observation operator
        H_0 = np.squeeze(H[:, :, i]).T

        # and the span of the filtered and unfiltered variables
        B_0f = np.squeeze(B[:, :ens_dim, i])
        B_0u = np.squeeze(B[:, ens_dim:, i])

        # the upper triangular matrix is loaded, and restricted to sub-blocks
        T_1 = np.squeeze(T[:, :, i])
        T_1pp = T_1[:ens_dim, :ens_dim]
        T_1pm = T_1[:ens_dim, ens_dim:]
        T_1mm = T_1[ens_dim:, ens_dim:]

        # define the square root of the filtered variables for the update
        [U, S_0, V] = np.linalg.svd(S_0sqr)
        S_0 = U @ np.diag(np.sqrt(S_0)) @ U.T

        # precision matrix
        H_f = H_0 @ B_0f
        H_u = H_0 @ B_0u
        Omega0 = S_0 @ H_f.T  @ np.linalg.inv(R) @ H_f @ S_0

        # reduced gain equation
        J_0 = S_0sqr @ H_f.T @ np.linalg.inv(H_f @ S_0sqr @ H_f.T + R)

        # UPDATE STEPS

        # filtered block update
        S_1sqr = T_1pp @ S_0 @ np.linalg.inv(I_f + Omega0) @ S_0 @ T_1pp.T + model_err * I_f
        if unfil_d == 1:
            tmp = (T_1pm - np.reshape(T_1pp @ J_0 @ H_u, [ens_dim, 1]))
            S_1sqr += P_0uu * tmp @ tmp.T
            Phi = T_1pp @ (I_f - J_0 @ H_f) @ P_0fu
            # reshape to get the exterior product of the two vectors and prevent unintended broadcasting
            Phi = np.reshape(Phi, [ens_dim, 1]) @ (T_1pm - np.reshape(T_1pp @ J_0 @ H_u, [ens_dim, 1])).T

            S_1sqr += Phi + Phi.T
        else:
            S_1sqr += (T_1pm - T_1pp @ J_0 @ H_u) @ P_0uu @ (T_1pm - T_1pp @ J_0 @ H_u).T
            Phi = T_1pp @ (I_f - J_0 @ H_f) @ P_0fu @ (T_1pm - T_1pp @ J_0 @ H_u).T
            S_1sqr += Phi + Phi.T

        # unfiltered update step
        if unfil_d == 1:
            P_1uu = model_err + P_0uu * T_1mm**2
        else:
            P_1uu = I_u * model_err + T_1mm @ P_0uu @ T_1mm.T

        # cross covariance steps
        if unfil_d == 1:
            P_1fu = (T_1pm - np.reshape(T_1pp @ J_0 @ H_u, [ens_dim, 1])) * P_0uu * T_1mm
            P_1fu += np.reshape(T_1pp @ (I_f - J_0 @ H_f) @ P_0fu * T_1mm, [ens_dim, 1])
        else:
            P_1fu = (T_1pm - T_1pp @ J_0 @ H_u) @ P_0uu @ T_1mm.T
            P_1fu += T_1pp @ (I_f - J_0 @ H_f) @ P_0fu @ T_1mm.T

        # broadcast the updates into the matrix P_traj
        P_traj[:ens_dim, :ens_dim, i + 1] = S_1sqr
        P_traj[:ens_dim, ens_dim:, i + 1] = P_1fu
        P_traj[ens_dim:, :ens_dim, i + 1] = P_1fu.T
        P_traj[ens_dim:, ens_dim:, i + 1] = P_1uu

    return P_traj


########################################################################################################################
# KFAUS forecast Riccati equation


def KFAUS_fore(T_kl, model_err, obs_err):
    """"This function returns a sequence of analysis error covariance matrices for the sqrt ricatti equation

    Note: we make the reduction in the calculation that R, Q are diagonal and time invariant.  T_k will be the upper
    triangular matrix in the QR factorization, it will be reduced to the appropriate block automatically."""

    [ens_dim, foo, nanl] = np.shape(T_kl)
    S_sqr_traj = np.zeros([ens_dim, ens_dim, nanl+1])

    S_sqr_traj[:, :, 0] = np.eye(ens_dim) * model_err
    Q = model_err * np.eye(ens_dim)
    R = obs_err * np.eye(ens_dim)
    I = np.eye(ens_dim)

    for i in range(nanl):
        # define the current operators for the ricatti equation
        T1 = np.squeeze(T_kl[:ens_dim, :ens_dim, i])
        S0_sqr = np.squeeze(S_sqr_traj[:, :, i])

        [U, S0, V] = np.linalg.svd(S0_sqr)
        S0 = U @ np.diag(np.sqrt(S0)) @ U.T

        Omega0 = S0 @ np.linalg.inv(R) @ S0
        S1_sqr = T1 @ S0 @ np.linalg.inv(I + Omega0) @ S0 @ T1.T + Q
        S_sqr_traj[:, :, i+1] = S1_sqr

    return S_sqr_traj

########################################################################################################################
# simulate AUS and corrected AUS


def simulate_kfAUS(B, T, H, ens_dim, model_err, obs_err):
    """This function simulates the parallel experiments of KFAUS and KFAUSE

    T_kl is the upper triangular matrices, from the QR factorizations of M_1, ..., M_nanl.  S_sqr_init will be the
    initialization of the prior error while model error and observational error are specified by constant scalar
    matrices; we assume S_0 is equivalent to the model error."""

    # NOTE: WE ALWAYS ASSUME THE INITIAL UNCERTAINTY TO BE THE MODEL ERROR
    # ####################################################################################################################
    # # KFAUS
    # P = KFAUS_fore(T_kl[:obs_dim, :obs_dim, :], model_err, obs_err)
    #
    # p_kfaus = []
    #
    # for i in range(nanl):
    #     # find the eigenvalues of the filtered covariance estimate
    #     tmp_p = np.linalg.eigvalsh(np.squeeze(P[:, :, i + 1]))
    #     p_kfaus.append(tmp_p[::-1])
    #
    # p_kfaus = np.array(p_kfaus)
    #
    # del P

    ####################################################################################################################
    # KFAUSE

    P = KFAUSE_fore(B, T, H, ens_dim, model_err, obs_err)
    p_kfause = analyze_p(P)

    return p_kfause
