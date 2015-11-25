## from pylab import *

import numpy as np
import scipy as sp
import scipy.integrate as spint
import matplotlib.pyplot as plt
import numpy.random as nprnd

F=8

def ComputeLorentzDrift(xin, tin):

    ## F = pars[0] ### don't know how to pass parameters to ODE solver
    n = np.size(xin, 0)
    # index for the next grid point
    nextk = np.mod(range(1,n+1), n)
    # index for the previous grid point
    prevk = np.mod(range(-1,n-1), n)
    # index for previous to previous grid point
    prevkk = np.mod(range(-1,n-1), n)
    drift = (xin[nextk] - xin[prevkk]) * xin[prevk] - xin + F;
    
    return drift


def IntegL96(tstep, h, t0, X0):
    
    N = np.size(X0, 0)
    X = np.zeros([N, tstep]); tm = np.zeros(tstep)
    X[:,0] = X0; tm[0] = t0
    h2 = h/2; h6 = h/6
    for i in range(tstep-1):
        currX = X[:, i]; currt = tm[i]
        tmid = currt + h2
        tend = tm[i+1] = currt + h
        k1 = ComputeLorentzDrift(currX,         currt);
        k2 = ComputeLorentzDrift(currX + h2*k1, tmid);
        k3 = ComputeLorentzDrift(currX + h2*k2, tmid);
        k4 = ComputeLorentzDrift(currX + h*k3,  tend);
        X[:,i+1] = currX + h6*(k1 +2*(k2+k3) + k4);

    return [X, tm]


xinit = 10*nprnd.randn(40)

### integrate using RK4
h = 0.001; tstep = 1000; tfin = tstep*h
[xrk4, ttmp] = IntegL96(tstep, h, 0, xinit)

### integrate using built-in ODE integrator
## pars[0] = F  ### don't know how to pass parameters to ODE solver
tgrid = np.linspace(0, tfin-h, tstep)
xsol = spint.odeint(ComputeLorentzDrift, xinit, tgrid)

plt.plot(tgrid,xsol[:,0], ttmp,xrk4[0,:])
plt.legend(["Builin-odeint", "RK4"])
plt.show()

