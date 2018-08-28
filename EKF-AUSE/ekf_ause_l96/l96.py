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
# Non-linear model vectorized for ensembles


def l96V(x, t, f):
    """"This describes the derivative for the non-linear Lorenz 96 Model of arbitrary dimension n.

    This will take the state vector x of shape sys_dim X ens_dim and return the equation for dxdt"""

    # shift minus and plus indices
    x_m_2 = np.concatenate([x[-2:, :], x[:-2, :]])
    x_m_1 = np.concatenate([x[-1:, :], x[:-1, :]])
    x_p_1 = np.concatenate([x[1:,:], np.reshape(x[0,:], [1, len(x[0, :])])], axis=0)

    dxdt = (x_p_1-x_m_2)*x_m_1 - x + f

    return dxdt

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
# non-linear L96 Runge Kutta


def l96_rk4_step(x, h, f):

    # calculate the evolution of x one step forward via RK-4
    k_x_1 = l96(x, h, f)
    k_x_2 = l96(x + k_x_1 * (h / 2.0), h, f)
    k_x_3 = l96(x + k_x_2 * (h / 2.0), h, f)
    k_x_4 = l96(x + k_x_3 * h, h, f)

    x_step = x + (h / 6.0) * (k_x_1 + 2 * k_x_2 + 2 * k_x_3 + k_x_4)

    return x_step

########################################################################################################################
# non-linear L96 Runge Kutta vectorized for ensembles


def l96_rk4_stepV(x, h, f):

    # calculate the evolution of x one step forward via RK-4
    k_x_1 = l96V(x, h, f)
    k_x_2 = l96V(x + k_x_1 * (h / 2.0), h, f)
    k_x_3 = l96V(x + k_x_2 * (h / 2.0), h, f)
    k_x_4 = l96V(x + k_x_3 * h, h, f)

    x_step = x + (h / 6.0) * (k_x_1 + 2 * k_x_2 + 2 * k_x_3 + k_x_4)

    return x_step


########################################################################################################################
# non-linear L96 2nd order Taylor Method


def l96_2tay_step(x, h, f):

    # calculate the evolution of x one step forward via the second order Taylor expansion

    # first derivative
    dx = l96(x, h, f)

    # second order taylor expansion
    x_step = x + dx * h + .5 * l96_jacobian(x) @ dx * h**2

    return x_step

########################################################################################################################
# 2nd order strong taylor SDE step
# Note: this method is from page 359, NUMERICAL SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS, KLOEDEN & PLATEN;
# this uses the approximate statonovich integrals defined on page 202


def l96_2tay_sde(x, h, f, diffusion, p, rho=False, alpha=False):
    # NOTE: recomputing rho and alpha can be redundant, but it is nice to keep the formula in the code.  Therefore, it
    # is not required to supply these constants, which depend on p, but they can be, outside of the step, for efficiency
    # in not re-computing these constants.  The discretization error depends loosely on p.  The upper bound on the error
    # of the approximate stratonovich integrals is order h**2*rho

    # infer system dimension
    sys_dim = len(x)

    # define the dxdt and the jacobian equations
    dx = l96(x, h, f)
    Jac_x = l96_jacobian(x)

    # we draw standard normal samples for all subsequent steps
    rndm = np.random.standard_normal([sys_dim, 2*p + 4])

    # NOTE: THIS NEEDS TO BE FIXED DUE TO THE FUNCTIONAL RELATIONSHIP BETWEEN W AND XI
    # we transform from the standard normal to the normal with variance given by the length of the time step
    W = rndm[:, 0] * np.sqrt(h)

    # we further define random variables to describe the fourier coefficients with which we approximate the strat ints
    xi = rndm[:, 1]
    mu = rndm[:, 2]
    phi = rndm[:, 3]

    # for each zeta and eta, we have sys_dim times p values
    zeta = rndm[:, 4: p+4]
    eta = rndm[:, p+4:]

    # we generate the random coefficients to simulate stratonovich integrals

    # start with auxiliary coefficients used to define the others --- these are defined once in terms of the order of
    # approximation p.  These can be supplied outside of the step for greater efficiency, though it is not required.
    if rho==0:
        rho = 1/12 - .5 * np.pi**(-2) * np.sum([1/(r**2) for r in range(1, p+1)])

    if alpha==0:
        alpha = (np.pi**2) / 180 - .5 * np.pi**(-2) * np.sum([1/r**4 for r in range(1, p+1)])

    # define the further coefficients
    a = np.zeros(sys_dim)
    b = np.zeros(sys_dim)
    for i in range(sys_dim):
        # define a and b vectors with length n
        a[i] = -(1/np.pi) * np.sqrt(2*h) * sum([zeta[i, r-1] / r for r in range(1, p+1)]) - 2*np.sqrt(h * rho)*mu[i]
        b[i] = np.sqrt(h*.5) * sum([eta[i, r-1] / r**2 for r in range(1, p+1)]) + np.sqrt(h*alpha) * phi[i]

    # vector of first order Stratonovich integrals
    V = .5 * (np.sqrt(h) * xi + a)

    # auxiliary functions

    # the triple stratonovich integral reduces in the lorenz 96 equation to a simple sum of the auxiliary functions, we
    # define these terms here abstractly so that we may efficiently compute the terms
    def C(l, j):
        C = np.zeros([p, p])

        # we will define the coefficient as a sum of matrix entries where r and k do not agree --- we compute this by a
        # set difference
        indx = set(range(1, p+1))

        for r in range(1, p+1):
            # vals are all values not equal to r
            vals = indx.difference([r])
            for k in vals:
                # and for row r, we define all columns to be given by
                C[r-1, k-1] = r/(r**2 - k**2) * (1/k * zeta[l, r-1] * zeta[j, k-1] - k/r * eta[l, r-1] * eta[j, k-1])

        # we return the sum of all values scaled by -1/2pi^2
        return -.5 * np.pi**(-2) * np.sum(C)

    def Psi(l, j):
        # psi will be a generic function of the indicies l and j, we will define psi plus and psi minus via this
        psi = .5 * h * a[l] * a[j] - (.5 / np.pi) * h**(1.5) * (xi[l]*b[j] + xi[j]*b[l])
        psi += .25 * h**(1.5) * (a[l]*xi[j] + a[j]*xi[l]) + (1/3) * h**2 * xi[l]*xi[j]
        psi -= h**2 * (C(l, j) + C(j, l))

        return psi

    # we define the approximations of the second order Stratonovich integral
    psi_plus = np.array([Psi((i-1) % sys_dim, (i+1) % sys_dim) for i in range(sys_dim)])
    psi_minus = np.array([Psi((i-2) % sys_dim, (i-1) % sys_dim) for i in range(sys_dim)])

    # CHECK ALL SIGNS IN THIS CODE LINE
    # the final vectorized step forward is given as
    x_step = x + dx * h + diffusion * W + 0.5 * Jac_x @ dx * h**2 + \
             h * diffusion * Jac_x @ V + diffusion**2 * (psi_plus - psi_minus)

    return x_step

########################################################################################################################
# Step the tangent linear model


def l96_step_TLM(x, Y, h, f, nonlinear_step):
    """"This function describes the step forward of the tangent linear model for Lorenz 96 via RK-4

    Input x is for the non-linear model evolution, while Y is the matrix of perturbations, h is defined to be the
    time step of the TLM.  This returns the forward non-linear evolution and the forward TLM evolution as
    [x_next,Y_next]"""

    h_mid = h/2

    # calculate the evolution of x to the midpoint
    x_mid = nonlinear_step(x, h_mid, f)

    # calculate x to the next time step
    x_next = nonlinear_step(x_mid, h_mid, f)

    k_y_1 = l96_jacobian(x).dot(Y)
    k_y_2 = l96_jacobian(x_mid).dot(Y + k_y_1 * (h / 2.0))
    k_y_3 = l96_jacobian(x_mid).dot(Y + k_y_2 * (h / 2.0))
    k_y_4 = l96_jacobian(x_next).dot(Y + k_y_3 * h)

    Y_next = Y + (h / 6.0) * (k_y_1 + 2 * k_y_2 + 2 * k_y_3 + k_y_4)

    return [x_next, Y_next]


########################################################################################################################
# Fundamental matrix solutions


def l96_tl_fms(x, tsteps, h, nanl, f, nonlinear_step):
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
            [x, Y] = l96_step_TLM(x, Y, h, f, nonlinear_step)

        fms[:, :, i] = Y[:]

    return [fms, x]

########################################################################################################################
