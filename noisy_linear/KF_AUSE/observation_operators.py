import numpy as np


########################################################################################################################

# random observational operator


def random_obs(sys_dim, obs_dim, nanl, seed):

    H = np.zeros([sys_dim, obs_dim, nanl])
    np.random.seed(seed)

    for i in range(nanl):
        tmp = np.random.randn(sys_dim, obs_dim)
        tmp = np.linalg.qr(tmp)
        H[:, :, i] = tmp[0]

    return H

########################################################################################################################
# alternating id obs


def alt_obs(sys_dim, obs_dim, nanl):
    """This function will define the standard observation operator for L96

    This will define a linear operator to give observations of alternating state components.  This will be used to
    define a lambda function for observations of the linear state components or their squares in the da routine."""

    H = np.eye(sys_dim)
    H_T = np.zeros([sys_dim, obs_dim, nanl])

    if sys_dim == obs_dim:
        H = np.eye(sys_dim)

    elif (obs_dim / sys_dim) > .5:
        # remove the trailing odd rows from the identity matrix
        R = int(sys_dim - obs_dim)
        H = np.delete(H, np.s_[-2*R::2], 0)

    elif (obs_dim / sys_dim) == .5:
        # remove odd rows
        H = np.delete(H, np.s_[1::2], 0)
    else:
        # remove odd rows and then the R trailing even rows
        H = np.delete(H, np.s_[1::2], 0)
        R = int(sys_dim/2 - obs_dim)
        H = np.delete(H, np.s_[-R:], 0)

    for i in range(nanl):
        H_T[:, :, i] = H.T

    return H_T

########################################################################################################################
# identity slice operator


def id_obs(sys_dim, obs_dim, nanl):
    H_I = np.zeros([sys_dim, obs_dim, nanl])
    for i in range(nanl):
        H_I[:, :, i] = np.eye(sys_dim, M=obs_dim)

    return H_I

########################################################################################################################
# dual subspace obs


def dual_subspace_obs(basis, obs_dim, normalized=True):
    """"This function will return the leading obs_dim dual vector observations

    To make an accurate comparison, we take the normalized SVD of the dual vectors with respect to the operator norm.
    We follow the general convention to return the matrix of column vectors, and transpose this matrix in the Riccati
    equation."""

    # infer dimensions
    [sys_dim, foo, nanl] = np.shape(basis)
    # define storage
    dual_obs = np.zeros([sys_dim, obs_dim, nanl])

    basis_s_val = np.zeros(obs_dim)
    dual_basis_s_val = np.zeros(obs_dim)

    for i in range(nanl):
        # we find the inverse matrix of the basis giving the row matrix of dual vectors
        basis_i = np.squeeze(basis[:, :, i])
        basis_inv = np.linalg.inv(basis_i)

        # we check how closely aligned the leading obs_dim columns of the basis are
        basis_i = basis_i[:, :obs_dim]
        nrm = np.linalg.norm(basis_i, ord=2)
        basis_i = basis_i / nrm
        [U, S, V] = np.linalg.svd(basis_i)
        basis_s_val = basis_s_val + S

        # we pass to the dual vectors of the leading obs_dim cols
        basis_inv = basis_inv[:obs_dim, :]

        if normalized:
            # take the operator norm
            nrm = np.linalg.norm(basis_inv, ord=2)

            # pass to the normalized matrix
            basis_inv = basis_inv / nrm

        # take the average singular values to check the alignment of the leading dual vectors
        [U, S, V] = np.linalg.svd(basis_inv)
        dual_basis_s_val = dual_basis_s_val + S

        dual_obs[:, :, i] = basis_inv.T

    basis_s_val = basis_s_val / nanl
    dual_basis_s_val = dual_basis_s_val / nanl

    return dual_obs, basis_s_val, dual_basis_s_val


########################################################################################################################
# pseudo inverse obs

def pseudo_inv_obs(basis, normalized=True):
    """This function will create an observational operator given by the pseudo inverse for a column basis for a subspace

    We follow all of the above conventions, and normalize with respect to the operator norm."""

    # infer dimensions
    [sys_dim, obs_dim, nanl] = np.shape(basis)
    # define the storage
    ps_inv_obs = np.zeros([sys_dim, obs_dim, nanl])

    for i in range(nanl):
        # define the pseudo inverse at one time step
        A_1d = np.squeeze(basis[:, :, i])
        A_plus = np.dot(A_1d.T, A_1d)
        A_plus = np.dot(np.linalg.inv(A_plus), A_1d.T)

        if normalized:
            # take the operator norm
            nrm = np.linalg.norm(A_plus, ord=2)

            # pass to the normalized matrix
            A_plus = A_plus / nrm

        ps_inv_obs[:, :, i] = A_plus.T

    return ps_inv_obs

########################################################################################################################
