# This is the module defining the propagator of linearized Lorenz 96
import numpy as np

########################################################################################################################
# Non-linear model


def l96(x, t, f):
    """"This describes the derivative for the non-linear Lorenz 96 Model of arbitrary dimension n.

    This will take the state vector x and return the equation for dxdt"""

    # shift minus and plus indices
    x_m_2 = np.concatenate([x[-2:], x[:-2]])
    x_m_1 = np.concatenate([x[-1:], x[:-1]])
    x_p_1 = np.append(x[1:], x[0])

    dxdt = (x_p_1-x_m_2)*x_m_1 - x + f

    return dxdt

########################################################################################################################
# non-linear L96


def l96_rk4_step(x, h, f):

    # calculate the evolution of x one step forward via RK-4
    k_x_1 = l96(x, h, f)
    k_x_2 = l96(x + k_x_1 * (h / 2.0), h, f)
    k_x_3 = l96(x + k_x_2 * (h / 2.0), h, f)
    k_x_4 = l96(x + k_x_3 * h, h, f)

    x_step = x + (h / 6.0) * (k_x_1 + 2 * k_x_2 + 2 * k_x_3 + k_x_4)

    return x_step


########################################################################################################################
# Jacobian


def l96_jacobian(x):
    """"This describes the Jacobian of the Lorenz 96, for arbitrary dimension, equation about the point x."""

    x_dim = len(x)

    dxF = np.zeros([x_dim, x_dim])

    for i in range(x_dim):
        i_m_2 = np.mod(i - 2, x_dim)
        i_m_1 = np.mod(i - 1, x_dim)
        i_p_1 = np.mod(i + 1, x_dim)

        dxF[i, i_m_2] = -x[i_m_1]
        dxF[i, i_m_1] = x[i_p_1] - x[i_m_2]
        dxF[i, i] = -1.0
        dxF[i, i_p_1] = x[i_m_1]

    return dxF

########################################################################################################################
# Step the tangent linear model


def l96_step_TLM(x, Y, h, f):
    """"This function describes the step forward of the tangent linear model for Lorenz 96 via RK-4

    Input x is for the non-linear model evolution, while Y is the matrix of perturbations, h is defined to be the
    time step of the TLM.  This returns the forward non-linear evolution and the forward TLM evolution as
    [x_next,Y_next]"""

    h_mid = h/2

    # calculate the evolution of x to the midpoint
    x_mid = l96_rk4_step(x, h_mid, f)

    # calculate x to the next time step
    x_next = l96_rk4_step(x_mid, h_mid, f)

    k_y_1 = l96_jacobian(x).dot(Y)
    k_y_2 = l96_jacobian(x_mid).dot(Y + k_y_1 * (h / 2.0))
    k_y_3 = l96_jacobian(x_mid).dot(Y + k_y_2 * (h / 2.0))
    k_y_4 = l96_jacobian(x_next).dot(Y + k_y_3 * h)

    Y_next = Y + (h / 6.0) * (k_y_1 + 2 * k_y_2 + 2 * k_y_3 + k_y_4)

    return [x_next, Y_next]


########################################################################################################################
# Fundamental matrix solutions


def l96_tl_fms(x, tsteps, h, nanl, f):
    """This function describes the matrix exponential for the equation of variations for a non-linear equation.

    The input x is the initial condition for model variables x_1, ..., x_n understood to be the variables of the fully
    nonlinear model.  The input tsteps is the number of steps to be taken between each initialization - note the
    the non-linear model will take twice this many steps.  The input h describes the step size of the tangent linear
    model, and nanl will denote the number number of analyses for the system, ie: the number of times we re-initialize
    the model.  The function returns the fundamental matrix solution of the tangent linear model along the trajectory
    defined by x along each time interval for the total number of analyses."""

    x_dim = len(x)
    fms = np.zeros([x_dim, x_dim, nanl])

    # compute the fundamental matrix solution for each time interval,
    for i in range(nanl):
        # initialize the tangent linear model with identity matrix and last point from non-linear run
        Y = np.eye(x_dim)

        # integrate the TLM  for tsteps and return the fundamental matrix solution to the next analysis time
        for j in range(tsteps):
            [x, Y] = l96_step_TLM(x, Y, h, f)

        fms[:, :, i] = Y[:]

    return [fms, x]

########################################################################################################################
