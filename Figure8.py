import Algorithms as alg
import numpy as np
#%% Topology of specific regions
N=100
VISp_meantf = 0.69

res = 100
X,Y = np.meshgrid(np.linspace(-5,5,res),np.linspace(-5,5,res))

#simulations
n_orients = 6
circle_size = np.zeros(n_orients)
mflds_VISp = np.zeros([5,n_orients])
mflds_VISrl = np.zeros(5)
mflds_VISl = np.zeros(5)
mflds_VISam = np.zeros(5)
n_perm = 40
distrib_size = [0]*n_orients
for q in range(n_orients):
    distrib_size[q] = np.zeros(n_perm)
    for p in range(n_perm):
        stim_temp_fqs = [8]
        stim_orients = np.linspace(0,2*np.pi,2**(q+2))
        repeats = int(128/2**(q+2))
        stim_N = repeats*len(stim_temp_fqs)*len(stim_orients)
        stims = np.zeros([stim_N,res,res])
        count = 0
        for i in range(repeats):
            for j in stim_temp_fqs:
                for k in range(len(stim_orients)):
                    stims[count,:,:] = alg.drifting_grating(X,Y,stim_orients[k],j/10)
                    count = count + 1
        
        neuron_orients = 2*np.pi*np.random.rand(N)#np.linspace(0,1*np.pi,N)#
        
        VISp_tf = np.random.exponential(VISp_meantf,N)
        
        VISp_filt = np.zeros([N,res,res])
        for i in range(N):
            VISp_filt[i,:,:] = np.random.exponential(20)*alg.gabor_filter(X,Y,neuron_orients[i],VISp_tf[i])
        
        VISp_pop = np.zeros([N,stim_N])
        for i in range(N):
            for j in range(stim_N):
                VISp_pop[i,j] = max(np.sum(np.multiply(VISp_filt[i],stims[j])),0)/res + 2*np.random.randn()
        
        hom_VISp = alg.normal_bd_dist(alg.full_hom_analysis(VISp_pop.T,metric='geodesic',R=-1,order=False,perm=None)[1])
        
        mflds_VISp[:,q] = mflds_VISp[:,q]+alg.check_mfld(hom_VISp[1],hom_VISp[2],thresh=0.2)#        
        distrib_size[q][p] = np.max(hom_VISp[1][:,1]-hom_VISp[1][:,0])
        circle_size[q] = circle_size[q] + np.max(hom_VISp[1][:,1]-hom_VISp[1][:,0])
    circle_size[q] = circle_size[q]/n_perm
    
distrib_size = np.vstack(distrib_size)
    
plt.figure(dpi=200)
plt.plot([2,3,4,5,6,7],circle_size,'k',linewidth=3)
plt.errorbar([2,3,4,5,6,7],circle_size,yerr = stats.sem(distrib_size,1),color='k')
plt.grid('on')
plt.xlabel('Stimulus density')
plt.ylabel('Persistence')
