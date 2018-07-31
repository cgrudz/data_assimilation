# this module contains functions for computing the lyapunov exponents and vectors, and bounds on the subspace for the
# analysis error covariance

import numpy as np

########################################################################################################################
# Backwards Lyapunov v's and e's


def l_back(M_kl, perts, save_lles=False, avg_trans=False, R_trans_hist=False, LV_hist=False,
           lead=0, rescale=1):
    """This function will estimate the backwards Lyapunov vectors and exponents via QR for a sequence of discrete maps

    The input M_kl will be understood as an array of size [n, n, k-l], containing the nxn propagators from time l to
    time k.  The input perts is the set of initial perturbations, the function returns an estimate of the leading
    dim(perts) backwards Lyapunov vectors and exponents at time k."""

    [sys_dim, foo, steps] = np.shape(M_kl)
    num_perts = len(perts[0, :])
    expos = np.zeros([sys_dim])

    if save_lles:
        LLEs = np.zeros([num_perts, steps])

    if avg_trans:
        exp_evo = np.zeros([num_perts, steps])

    if R_trans_hist:
        R_hist = np.zeros([sys_dim, foo, steps])

    if LV_hist:
        LV_evo = np.zeros([sys_dim, num_perts, steps - lead])

    for i in range(steps):
        # propagate
        perts = M_kl[:, :, i].dot(perts)

        # QR and store the logs of the diagonal of R
        [perts, R] = np.linalg.qr(perts, mode='complete')
        R_diag = np.log(np.abs(np.diagonal(R)))

        # store LLEs
        if save_lles:
            # we compute the localy lyapunov exponents with an optional rescaling factor to account for the interval
            # length of the discretization
            LLEs[:, i] = R_diag[:]/rescale

        expos = expos + R_diag

        # R history
        if R_trans_hist:
            R_hist[:, :, i] = R

        # store a history of the average transient growth rates in the current run
        if avg_trans:
            exp_evo[:, i] = expos[:] / ((i + 1) * rescale)

        if i >= lead:
            if LV_hist:
                LV_evo[:, :, i - lead] = perts

    # the average over the discretization interval
    expos /= (steps * rescale)

    # define the output dict
    output = {}

    output['perts'] = perts
    output['expos'] = expos
    
    if save_lles:
        output['lles'] = LLEs

    if avg_trans:
        output['expo_evo'] = exp_evo

    if R_trans_hist:
        output['R_hist'] = R_hist

    if LV_hist:
        output['BLVs'] = LV_evo

    return output

########################################################################################################################
# Forwards Lyapunov v's and e's


def l_for(M_kl, perts, expos_init, save_lles=False, avg_trans=False, R_trans_hist=False, LV_hist=False, lead=0,
          rescale=1):
    """This function will estimate the forwards Lyapunov vectors and exponents via QR for a sequence of discrete maps

    The input M_kl will be understood as an array of size [n, n, k-l], containing the nxn propagators from time l to
    time k.  The input perts is the set of initial perturbations, the function returns an estimate of the leading
    dim(perts) forwards Lyapunov vectors and exponents at time k."""

    [sys_dim, foo, steps] = np.shape(M_kl)
    num_perts = len(perts[0, :])
    expos = expos_init

    if save_lles:
        LLEs = np.zeros([num_perts, steps])

    if avg_trans:
        exp_evo = np.zeros([num_perts, steps])

    if R_trans_hist:
        R_hist = np.zeros([sys_dim, foo, steps])

    if LV_hist:
        LV_evo = np.zeros([sys_dim, num_perts, steps - lead])

    for i in range(steps):
        # propagate
        tmp = np.squeeze(M_kl[:, :, steps - 1 - i]).transpose()
        perts = tmp.dot(perts)

        # QR and store the logs of the diagonal of R
        [perts, R] = np.linalg.qr(perts, mode='complete')
        R_diag = np.log(np.abs(np.diagonal(R)))
        expos = expos + R_diag

        # store LLEs
        if save_lles:
            LLEs[:, steps - 1 - i] = R_diag[:]/rescale

        # R history
        if R_trans_hist:
            R_hist[:, :, steps - 1 - i] = R

        # store a history of the average transient growth rates in the current run
        if avg_trans:
            exp_evo[:, steps - 1 - i] = expos[:] / ((i + 1) * rescale)

        if i >= lead:
            if LV_hist:
                LV_evo[:, :, steps - 1 - i] = perts

    # we compute both the average value for the exponents and the raw sum for batch processing sequences of matrices
    # and their associated exponents over the sequence of all batches

    # raw sum of exponents for batch processing
    expos_sum = np.empty_like(expos)
    expos_sum[:] = expos

    # the average value over the current run
    expos /= (steps * rescale)

    # define the output list, by default returning the vectors and exponents
    output = [perts, expos, expos_sum]

    if save_lles:
        output.append(LLEs)

    if avg_trans:
        output.append(exp_evo)

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