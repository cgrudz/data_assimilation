import numpy as np

########################################################################################################################
# alternating id obs


def alt_obs(sys_dim, obs_dim):
    """This function will define the standard observation operator for L96

    This will define a linear operator to give observations of alternating state components.  This will be used to
    define a lambda function for observations of the linear state components or their squares in the da routine."""

    H = np.eye(sys_dim)

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

    return H

########################################################################################################################
# return sequence of observations


def obs_seq(truth, obs_dim, obs_un, obs_op, seed):
    """This function takes the true state trajectory, and parameters, and returns an associated sequence of observations
    """

    np.random.seed(seed)
    [sys_dim, nanl] = np.shape(truth)
    obs = np.zeros([obs_dim, nanl])
    zero = np.zeros(obs_dim)
    R = np.eye(obs_dim) * obs_un

    for j in range(nanl):
        obs[:, j] = obs_op @ np.squeeze(truth[:, j]) + np.random.multivariate_normal(zero, R)

    return obs

########################################################################################################################
