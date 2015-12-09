import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def distribution_plt_NRS(pdf_series,obs,directory,exp_num):
    # this function plots the series of prior and posterior distributions
    # in the dictionary pdf_series and saves the figures in the directory
    # supplied by the string `directory'

    directory = directory + 'NRS_filter/exp_num_' + str(exp_num) + '/'
    A = 'A_'
    keys = len(pdf_series.keys())
    for i in range(keys):
        key = A + str(i)
        temp  = plt.figure()
        cloud = pdf_series[key]['prior']
        # compute the mean of the prior
        weights = pdf_series[key]['prior_weight']
        remainder = len(weights)
        plt.title('Prior: '+str(i)+' Particles: '+str(remainder))
        mean = np.sum(cloud.transpose()*weights,1)
    
        plt.scatter(cloud[:,0],cloud[:,1],c=weights)
        plt.plot(obs[i,0],obs[i,1],'rs')
        plt.plot(mean[0],mean[1],'ys')
        plt.axis([-.3,1.8,-1.8,1])
        plt.savefig(directory+key+'a.png')
        plt.close(temp)
    
        temp  = plt.figure()
        cloud = pdf_series[key]['post']
        # compute the mean of the posterior
        weights = pdf_series[key]['post_weight']
        remainder = len(weights)
        plt.title('Post: '+str(i)+' Particles: '+str(remainder))
        mean = np.sum(cloud.transpose()*weights,1)
    
        plt.scatter(cloud[:,0],cloud[:,1],c=weights)
        plt.plot(obs[i,0],obs[i,1],'rs')
        plt.plot(mean[0],mean[1],'ys')
        plt.axis([-.3,1.8,-1.8,1])
        plt.savefig(directory+key+'b.png')
        plt.close(temp)
