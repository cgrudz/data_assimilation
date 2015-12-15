import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plt_NRS(pdf_series,obs,directory,exp_num):
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
        plt.plot(obs[i,0],obs[i,1],'rD')
        plt.plot(mean[0],mean[1],'yD')
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
        plt.plot(obs[i,0],obs[i,1],'rD')
        plt.plot(mean[0],mean[1],'yD')
        plt.axis([-.3,1.8,-1.8,1])
        plt.savefig(directory+key+'b.png')
        plt.close(temp)

def plt_SIR(pdf_series,obs,directory,exp_num):
    # this function plots the series of prior and posterior distributions
    # in the dictionary pdf_series and saves the figures in the directory
    # supplied by the string `directory'
    
    directory = directory + 'SIR_filter/exp_num_' + str(exp_num) + '/'
    A = 'A_'
    keys = len(pdf_series.keys())
    for i in range(keys):
        key = A + str(i)
        temp  = plt.figure()
        cloud = pdf_series[key]['prior']
        # compute the mean of the prior
        weights = pdf_series[key]['prior_weight']
        resample = pdf_series[key]['prior_resample']
        plt.title('Prior: '+str(i)+' Resamples: '+str(resample))
        mean = np.sum(cloud.transpose()*weights,1)
    
        plt.scatter(cloud[:,0],cloud[:,1],c=weights)
        plt.plot(obs[i,0],obs[i,1],'rD')
        plt.plot(mean[0],mean[1],'yD')
        plt.axis([-.3,1.8,-1.8,1])
        plt.savefig(directory+key+'a.png')
        plt.close(temp)
    
        temp  = plt.figure()
        cloud = pdf_series[key]['post']
        # compute the mean of the posterior
        weights = pdf_series[key]['post_weight']
        resample = pdf_series[key]['post_resample']
        plt.title('Post: '+str(i)+' Resamples: '+str(resample))
        mean = np.sum(cloud.transpose()*weights,1)
    
        plt.scatter(cloud[:,0],cloud[:,1],c=weights)
        plt.plot(obs[i,0],obs[i,1],'rD')
        plt.plot(mean[0],mean[1],'yD')
        plt.axis([-.3,1.8,-1.8,1])
        plt.savefig(directory+key+'b.png')
        plt.close(temp)

def plt_EnKF(pdf_series,ens_size,obs,directory,exp_num):
    # this function plots the series of prior and posterior distributions
    # in the dictionary pdf_series and saves the figures in the directory
    # supplied by the string `directory'
    
    directory = directory + 'EnKF_filter/exp_num_' + str(exp_num) + '/'
    A = 'A_'
    keys = len(pdf_series.keys())
    likelyhood = np.ones(ens_size)/float(ens_size)
    for i in range(keys):
        key = A + str(i)

        # plot prior distribution
        temp  = plt.figure()
        cloud = pdf_series[key]['prior']
        plt.title('Prior: '+str(i))
        mean = np.mean(cloud,0)
        
        plt.scatter(cloud[:,0],cloud[:,1],c=likelyhood)
        plt.plot(obs[i,0],obs[i,1],'cD')
        plt.plot(mean[0],mean[1],'yD')
        plt.axis([-1.5,2.5,-2.5,1.5])
        plt.savefig(directory+key+'a.png')
        plt.close(temp)
    
        # plot posterior distribution
        temp  = plt.figure()
        cloud = pdf_series[key]['post']
        plt.title('Post: '+str(i))
        mean = np.mean(cloud,0)
        cov = np.cov(cloud.transpose())
        
        # vectorize the innovation from the integration form
        innov = -1*(cloud - mean).transpose()
        # compute the exponent of the likelyhood function
        l_hood  = np.sum(cov.dot(innov)*innov,axis=0)
        l_hood = np.exp(-0.5*l_hood)**(1.0/3.0)
        likelyhood = np.ones(ens_size)*l_hood
        likelyhood = likelyhood/np.sum(likelyhood)
        
        plt.scatter(cloud[:,0],cloud[:,1],c=likelyhood)
        plt.plot(obs[i,0],obs[i,1],'cD')
        plt.axis([-1.5,2.5,-2.5,1.5])
        plt.savefig(directory+key+'b.png')
        plt.close(temp)
