import numpy as np

def resample_systematic(w):

    # This function takes an array of weights and returns an array of 
    # indices - the indices map the array of particles to the re-sampled
    # particles, with each index corresponding to a copy of the associated
    # particle in the original array.

    N   = len(w)
    ind = np.zeros(N)
    N   = float(N)
    
    Q = np.cumsum(w)

    T = np.linspace(0,1-1/N,N) + np.random.rand()/N
    
    i = 0
    j = 0

    while i < N:
        if T[i] < Q[j]:
            ind[i]=j
            i=i+1
        else:
            j=j+1        

    return ind
