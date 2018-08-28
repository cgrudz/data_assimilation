# this module contains functions for computing the lyapunov exponents and vectors, and bounds on the subspace for the
# analysis error covariance

import numpy as np
from sympy import Matrix

########################################################################################################################
# Backwards Lyapunov v's and e's


def l_back(M_kl, perts, expos, R_trans_hist=False, LV_hist=False):
    """This function will estimate the backwards Lyapunov vectors and exponents via QR for a sequence of discrete maps

    The input M_kl will be understood as an array of size [n, n, k-l], containing the nxn propagators from time l to
    time k.  The input perts is the set of initial perturbations, the function returns an estimate of the leading
    dim(perts) backwards Lyapunov vectors and exponents at time k."""

    [sys_dim, foo, steps] = np.shape(M_kl)
    num_perts = len(perts[0, :])

    if R_trans_hist:
        R_hist = np.zeros([sys_dim, foo, steps])

    if LV_hist:
        LV_evo = np.zeros([sys_dim, num_perts, steps + 1])
        LV_evo[:, :, 0] = perts

    for i in range(steps):
        # propagate
        perts = M_kl[:, :, i].dot(perts)

        # QR and store the logs of the diagonal of R
        [perts, R] = np.linalg.qr(perts, mode='complete')
        R_diag = np.log(np.abs(np.diagonal(R)))
        expos = expos + R_diag

        # R history
        if R_trans_hist:
            R_hist[:, :, i] = R

        if LV_hist:
            LV_evo[:, :, i + 1] = perts

    # define the output list, by default returning the vectors and exponents
    output = [perts, expos]

    if R_trans_hist:
        output.append(R_hist)

    if LV_hist:
        output.append(LV_evo)

    return output

########################################################################################################################
# Forwards Lyapunov v's and e's


def l_for(M_kl, perts, expos, R_trans_hist=False, LV_hist=False):
    """This function will estimate the forwards Lyapunov vectors and exponents via QR for a sequence of discrete maps

    The input M_kl will be understood as an array of size [n, n, k-l], containing the nxn propagators from time l to
    time k.  The input perts is the set of initial perturbations, the function returns an estimate of the leading
    dim(perts) forwards Lyapunov vectors and exponents at time k."""

    [sys_dim, foo, steps] = np.shape(M_kl)
    num_perts = len(perts[0, :])

    if R_trans_hist:
        R_hist = np.zeros([sys_dim, foo, steps])

    if LV_hist:
        # note we must take the initial perts as the flvs at the end time, these recursively compute the flvs at time
        # steps back in time
        LV_evo = np.zeros([sys_dim, num_perts, steps + 1])
        LV_evo[:, :, -1] = perts

    for i in range(steps):
        # propagate
        tmp = np.squeeze(M_kl[:, :, steps - 1 - i]).transpose()
        perts = tmp.dot(perts)

        # QR and store the logs of the diagonal of R
        [perts, R] = np.linalg.qr(perts, mode='complete')
        R_diag = np.log(np.abs(np.diagonal(R)))
        expos = expos + R_diag

        # R history
        if R_trans_hist:
            R_hist[:, :, steps - 1 - i] = R

        # each loop produces the flvs one step back in time
        if LV_hist:
            LV_evo[:, :, steps - 1 - i] = perts

    # define the output list, by default returning the vectors and exponents
    output = [perts, expos]

    if R_trans_hist:
        output.append(R_hist)

    if LV_hist:
        output.append(LV_evo)

    return output

########################################################################################################################
# bound on the stable subspace


def stable_bound(transients):
    """ This function computes the bound for the analysis covariance on the stable backwwards Lyapunov directions

    It is assumed that transients is an array of size [n - n_0, n - n_0,num_steps], where n-n_0 is the number of stable
    exponents for the system.  This is to be the sequence of the lower right block of the QR factorizations of the
    system propagator.  The return is the sequence of bounds, and the local transient rates."""

    [stab_dim, foo, steps] = np.shape(transients)

    # note that the bound is always bounded below by 1, as the latest term is given by the identity matrix
    cum_bnd = np.ones([stab_dim, steps])

    for i in range(steps):
        # recursively compute the product of the transients from current time back to zero
        theta_ij = np.eye(stab_dim)
        for j in range(i+1):
            temp = np.squeeze(transients[:, :, i-j])
            theta_ij = theta_ij.dot(temp)

            # compute the norm square of the row
            temp = np.sum(theta_ij*theta_ij, axis=1)

            # if the computation becomes trivial stop
            if np.all(temp < 1e-15):
                break
            cum_bnd[:, i] += temp

    return cum_bnd

########################################################################################################################
# Function for computing the row norm giving instantaneous growth of the LV


def row_norm(trans):
    """ This function computes the row norms of the transient R matrices, yielding the behavior of the orthogonal vects

    This function likewise takes the transients array and will simply store the array of associated norms for each time
    step."""

    [state_dim, foo, steps] = np.shape(trans)

    # storage for the LLE's
    LLE = np.zeros([state_dim, steps])
    
    for i in range(steps):
        # for each step compute the norm on each row
        for j in range(state_dim):
            temp = np.squeeze(trans[j,:,i])
            LLE[j,i] = np.sqrt(temp.dot(temp))

    return LLE

########################################################################################################################
# Compute CLVs one step


def l_covs(BLVs, FLVs):
    """This function takes an array of backward/ forward Lyapunov vectors for one step and returns the associated CLVs

    The computation is performed by the method of LU factorization, requiring a full basis of backward/ forward Lyapunov
    vectors, though allows for spectral degeneracy.  In this case, repeat CLVs will be returned."""

    # infer the size of the model
    [sys_dim, foo] = np.shape(FLVs)
    # define the P matrix as in Kuptsov & Parlitz
    P = FLVs.T.dot(BLVs)

    # Define the upper triangular matrix for the symmetry relationship with the CLVs
    A_minus = np.zeros([sys_dim, sys_dim])

    for j in range(1, sys_dim+1):
        # for each rectangular sub matrix P[0:j-1, 0:j] we compute the null space, which yields the non-zero elements of
        # of the jth column of the upper triangular matrix A_minus
        P_jj = Matrix(P[0:j-1, 0:j])
        A_minus_j = P_jj.nullspace()
        # in case of degeneracy of the spectrum, we arbitrarily pick the first vector in the nullspace
        A_minus_j = np.squeeze(np.array(A_minus_j[0]))

        # we fill the non-zero entries of column j of A_minus
        A_minus[:j, j-1] = A_minus_j

    # we define the arbitrary CLVs via the QR factorization
    CLVs = BLVs.dot(A_minus)

    # and we impose the norm one condition
    for i in range(sys_dim):
        clv_i = np.squeeze(CLVs[:, i])
        clv_i = clv_i/np.sqrt(np.dot(clv_i, clv_i))

        CLVs[:, i] = clv_i[:]

    return CLVs

########################################################################################################################
# Compute CLVs over range


def l_covs_range(BLVs, FLVs):
    """This function takes a range of FLVs and BLVs and returns the associated CVLs at each time step in range.

    This is a simple wrapper of the clv function in which we call this function for each matching time step in an array
    of FLVs/ CLVs over the same time steps."""

    # we take the shape of the BLVs to be sys_dim x sys_dim x nanl
    [sys_dim, foo, nanl] = np.shape(BLVs)

    # define storage
    CLVs = np.zeros([sys_dim, sys_dim, nanl])

    for i in range(nanl):
        blv = np.squeeze(BLVs[:, :, i])
        flv = np.squeeze(FLVs[:, :, i])
        clv = l_covs(flv, blv)

        CLVs[:, :, i] = clv[:]

    return CLVs