import numpy as np
from pylab import find

def js_div(P,P_weight,Q,Q_weight):
    """This is the Jensen-Shannon Divergence between two weighted discrete distributions
    
    Function takes distribution P with weights P_weights, and likewise for Q.  We assume that
    the particle cloud P/Q is of dimension particle_number X statedimension."""
    
    # Merge the pdf's into mixture pdfs M.  Computing divergence of P||M and Q||M
    # we may negelect the points where P(z)=0, Q(z)=0 respectively, so that we create two
    # mixture pdfs for the comparison, regarding only points in P and Q respectively.
    M_p_w = P_weight*.5
    M_q_w = Q_weight*.5
    
    q_ens = len(Q[:,1])
    for i in range(q_ens):
        # find the row index of matching points
        indx = find((P==Q[i,:])[:,0])
        if np.size(indx)!= 0:
            # the position is shared between to the two pdfs, and therefore, we will append
            # the weight information, by summing .5 the Q weight with .5 the P weight 
            M_p_w[indx] = M_p_w[indx] + .5*Q_weight[i]
            M_q_w[i] = M_q_w[i] + .5*P_weight[indx]

    # compute the Klullback-Leibler divergence of P WRT M and Q WRT M
    P_div = P_weight*np.log(P_weight/M_p_w)
    P_div = sum(P_div)
    
    Q_div = Q_weight*np.log(Q_weight/M_q_w)
    Q_div = sum(Q_div)
    
    div = .5*P_div + .5*P_div
    
    return(div)