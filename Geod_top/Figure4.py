import os
from allensdk.brain_observatory.ecephys.ecephys_session import (
    EcephysSession)
import Algorithms as alg
from joblib import Parallel, delayed
from scipy.stats import percentileofscore, sem
import numpy as np

cdir = '/home/kosio/Master_thesis/Data/raw_data/session_datasets'
datasets = np.sort(os.listdir(cdir))

alpha=0.05

pdiags = []
pdiags_l2 = []

exp_var = []
dmat_geod = []
dmat_l2 = []
count =  0
all_data = []
region_list = [[]]*5
for session_id in datasets:
    nwb_path = '/home/kosio/Master_thesis/Data/raw_data/session_datasets/' + str(session_id)
#
    session = EcephysSession.from_nwb_path(nwb_path, api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf
    })
    print(session_id)

    vis_regions = ['LP','VISp','VISrl','VISl','VISam']
    stim_dat = {}
    trial_dat = {}
    pcloud_dg = {}
    num_bins_dg = 2#0.25#
    for i in vis_regions:
        temp_data = alg.open_data(session,'drifting_gratings',i,num_bins_dg,bin_strt=0,frate=1,snr=2)
        stim_dat[str(i)+' dg'] = temp_data[0]
        pcloud_dg[str(i)] = np.vstack(temp_data[0])
    
    region_ids = []
    count_region = 0
    pcloud_mean_dg = {}
    for i in vis_regions:
        pcloud_mean_dg[str(i)] = []
        for j in range(len(stim_dat[str(i)+' dg'])):
                pcloud_mean_dg[str(i)].append(stim_dat[str(i)+' dg'][j])#[:,region_tun_curves[count]])
        if len(np.squeeze(np.vstack(pcloud_mean_dg[str(i)]))[0,:])>0:
            pcloud_mean_dg[str(i)] = np.squeeze(np.vstack(pcloud_mean_dg[str(i)]))
            region_ids.append(count_region)
            region_list[count_region] = region_list[count_region]+[len(all_data)]
            all_data.append(pcloud_mean_dg[str(i)])
        else:
            del pcloud_mean_dg[str(i)]
        count_region = count_region+1
    vis_regions = list(pcloud_mean_dg.keys())


strength = np.ones(len(all_data))
strength2 = np.ones(len(all_data))
strength_l2 = np.ones(len(all_data))
strength2_l2 = np.ones(len(all_data))

significance = np.ones(len(all_data))#np.ones([5,len(datasets)])
significance2 = np.ones(len(all_data))#np.ones([5,len(datasets)])
significance_l2 = np.ones(len(all_data))#np.ones([5,len(datasets)])
significance2_l2 = np.ones(len(all_data))#np.ones([5,len(datasets)])

p_imager = pimg.PersistenceImager(pixel_size=0.1,kernel_params={'sigma':np.array([[0.05,0],[0,0.05]])})
p_imager.weight = pimg.weighting_fxns.persistence
p_imager.weight_params = {'n': 2}


geod_params = [0.005,0.05,0.1,0.2,0.5,1,2]
mean_d_error, mean_p_error = [0]*(len(all_data)*len(geod_params)), [0]*(len(all_data)*len(geod_params))
mean_d_sem, mean_p_sem = [0]*(len(all_data)*len(geod_params)), [0]*(len(all_data)*len(geod_params))
pimg_geod, pimg_l2 = [], []
sign_h1_ratio = []
sign_h1_ratio_l2 = []
percentile_geod = np.zeros(len(all_data))
percentile_geod2 = np.zeros(len(all_data))

h1_betti = np.zeros(len(all_data))
h2_betti = np.zeros(len(all_data))
for g in geod_params:
    count=0
    n_permutations = 1
    max_hom_dg1 = np.zeros([n_permutations,len(all_data)])
    max_hom_dg1_l2 = np.zeros([n_permutations,len(all_data)])
    max_hom_dg2 = np.zeros([n_permutations,len(all_data)])
    max_hom_dg2_l2 = np.zeros([n_permutations,len(all_data)])
    for d in all_data:    
        exp_var.append(PCA().fit(d).explained_variance_ratio_)
        dmat_temp = alg.geodesic(d,eps=g)
        dmat_geod.append(dmat_temp)
        hom_dg_temp = alg.normal_bd_dist(tda(dmat_temp,distance_matrix=True,maxdim=1,n_perm=None)['dgms'])
        temp_pimg_geod = p_imager.transform(hom_dg_temp[1])
        pimg_geod.append(temp_pimg_geod/np.max(temp_pimg_geod))
        if g == geod_params[0]:
            dmat_temp = alg.pairwise_distances(d)
            hom_dg_temp_l2 = alg.normal_bd_dist(tda(dmat_temp,distance_matrix=True,maxdim=1,n_perm=None)['dgms']) 
            dmat_l2.append(dmat_temp)
            strength_l2[count] = alg.max_pers(hom_dg_temp_l2)
            pdiags_l2.append(hom_dg_temp_l2)
            temp_pimg_l2 = p_imager.transform(hom_dg_temp_l2[1])
            pimg_l2.append(temp_pimg_l2/np.max(temp_pimg_l2))
        strength[count] = alg.max_pers(hom_dg_temp)
        pdiags.append(hom_dg_temp)
        n_nrns = len(d.T)
        n_times = len(d)
        perm_analysis = np.asanyarray(Parallel(n_jobs=20,backend='loky')(delayed(alg.full_perm_analysis)(d,dim=1,geod_ep=g) for p in range(n_permutations)))
        max_hom_dg1[:,count] = perm_analysis
        temp_frates = np.sum(d,0)/n_times
        if g == geod_params[0]:
            perm_analysis_l2 = np.asanyarray(Parallel(n_jobs=20,backend='loky')(delayed(alg.full_perm_analysis)(d,dim=1,geod_ep=g,metric='LP') for p in range(n_permutations)))
            max_hom_dg1_l2[:,count] = perm_analysis_l2
            significance_l2[count] = 1-percentileofscore(max_hom_dg1_l2[:,count],strength_l2[count])/100
#        max_hom_dg1[j,count] = max_pers(hom_dg_temp)
#            max_hom_dg2[j,count] = max_pers(hom_dg_temp,dim=2)
#            max_hom_dg2_l2[j,i] = max_pers(hom_dg_temp_l2,dim=2)
        percentile_geod[count] = np.percentile(max_hom_dg1[:,count],100*(1-alpha/len(all_data)))
#        percentile_geod2[count] = np.percentile(max_hom_dg2[:,count],100*(1-alpha/len(all_data)))
        h1_betti[count] = np.sum((pdiags[count][1][:,1]-pdiags[count][1][:,0])>percentile_geod[count])
#        h2_betti[count] = np.sum((pdiags[count][2][:,1]-pdiags[count][2][:,0])>percentile_geod2[count])
        significance[count] = 1-percentileofscore(max_hom_dg1[:,count],strength[count])/100
#        significance2[count] = 1-percentileofscore(max_hom_dg2[:,count],strength2[count])/100
    #        significance2_l2[region_ids[i],count] = 1-percentileofscore(max_hom_dg2_l2[:,i],strength2_l2[i,count])/100
        count = count + 1
        print(count)
    significance_fdr = alg.fdr(significance)
    significancel2_fdr = alg.fdr(significance_l2)    
    sign_h1_ratio.append(np.sum(significance_fdr<alpha)/len(all_data))
    sign_h1_ratio_l2.append(np.sum(significancel2_fdr<alpha)/len(all_data))

mean_derror = np.zeros([len(all_data),len(geod_params)])
mean_perror = np.zeros([len(all_data),len(geod_params)])
sem_derror = np.zeros([len(all_data),len(geod_params)])
sem_perror = np.zeros([len(all_data),len(geod_params)])
count = 0
for g in range(len(geod_params)):
    for i in range(len(all_data)):
        distance_error = np.sqrt((dmat_l2[i]-dmat_geod[count])**2)
        mean_derror[i,g] = np.mean(distance_error)
        sem_derror[i,g] = sem(distance_error.flatten())
        pimg_error = np.sqrt((pimg_geod[count]-pimg_l2[i])**2)
        mean_perror[i,g] = np.mean(pimg_error)
        sem_perror[i,g] = sem(pimg_error.flatten())   
        count = count + 1
mean_derror_tot = np.mean(mean_derror,0)
mean_perror_tot = np.mean(mean_perror,0)
mean_dsem_tot = sem(mean_derror,0)
mean_psem_tot = sem(mean_perror,0)

valid_cases = np.where(strength.flatten()!=1)[0]
sig_both = np.where(np.logical_and(significance_fdr.flatten()<alpha,significancel2_fdr.flatten()<alpha))[0]
sig_geod = np.where(np.logical_and(significance_fdr.flatten()<alpha,significancel2_fdr.flatten()>alpha))[0]
sig_l2 = np.where(np.logical_and(significance_fdr.flatten()>alpha,significancel2_fdr.flatten()<alpha))[0]
sig_none = np.where(np.logical_and(significance_fdr.flatten()>alpha,significancel2_fdr.flatten()>alpha))[0]
#%%
samples = len(pcloud_dg)
geod_params = np.logspace(-2,1,50) #50 values
nums = len(geod_params)

hom_geod = [0]*nums
distance_error = [0]*nums
mean_derror, sem_derror = np.zeros(nums), np.zeros(nums)
S1 = np.copy(pcloud_dg)
d_l2, hom_l2 = alg.full_hom_analysis(S1,metric='LP',order=False,perm=None,dim=1)
hom_l2 = alg.normal_bd_dist(hom_l2)
d_geod, hom_geod = [0]*nums, [0]*nums
pimg_geod, pimg_l2 = [0]*nums, [0]*nums
pimg_error = [0]*nums
mean_perror, sem_perror = np.zeros(nums), np.zeros(nums)
p_imager = pimg.PersistenceImager(pixel_size=0.1,kernel_params={'sigma':np.array([[0.01,0],[0,0.01]])})
p_imager.weight = pimg.weighting_fxns.persistence
p_imager.weight_params = {'n': 2}
pimg_l2 = p_imager.transform(hom_l2[1])
for i in range(nums):
    d_geod[i], hom_geod[i] = alg.full_hom_analysis(S1,metric='geodesic',order=False,perm=None,dim=1,R=-1,Eps=geod_params[i])
    hom_geod[i] = alg.normal_bd_dist(hom_geod[i])
    distance_error[i] = np.sqrt((d_l2-d_geod[i])**2)
    mean_derror[i] = np.mean(distance_error[i])
    sem_derror[i] = sem(distance_error[i].flatten())
    pimg_geod[i]  = p_imager.transform(hom_geod[i][1])
    pimg_error[i] = np.sqrt((pimg_geod[i]-pimg_l2)**2)
    mean_perror[i] = np.mean(pimg_error[i])
    sem_perror[i] = sem(pimg_error[i].flatten())    

#%%
valid_cases = np.where(strength.flatten()!=1)[0]
vis_regions = ['LP','VISp','VISrl','VISl','VISam']

plt.figure(dpi=200)
plt.plot(geod_params,sign_h1_ratio,'g-o')
plt.plot(geod_params,sign_h1_ratio_l2,'r')
plt.grid('on')
plt.xscale('log')

fig, ax1 = plt.subplots(dpi=200)
plt.grid('on')
ax1.errorbar(geod_params,mean_derror_tot,yerr=mean_dsem_tot,color='k')
ax1.set_xlabel('Neighborhood size parameter')
ax1.set_ylabel('Distance difference')
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.errorbar(geod_params,mean_perror_tot,yerr=mean_psem_tot,color='r')
ax2.set_ylabel('Persistence image difference',color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.tight_layout()


plt.figure(dpi=200)
plt.plot(strength_l2.flatten()[sig_none],strength.flatten()[sig_none],'m.')
plt.plot(strength_l2.flatten()[sig_both],strength.flatten()[sig_both],'b.')
plt.plot(strength_l2.flatten()[sig_l2],strength.flatten()[sig_l2],'r.')
plt.plot(strength_l2.flatten()[sig_geod],strength.flatten()[sig_geod],'g.')
plt.plot([0,.5],[0,.5],'k')
plt.xlabel('Euclidean')
plt.ylabel('Geodesic')
plt.tight_layout()

geod_region = [strength.flatten()[region_list[0]],strength.flatten()[region_list[1]],strength.flatten()[region_list[2]],strength.flatten()[region_list[3]],strength.flatten()[region_list[4]]]
l2_region = [strength_l2.flatten()[region_list[0]],strength_l2.flatten()[region_list[1]],strength_l2.flatten()[region_list[2]],strength_l2.flatten()[region_list[3]],strength_l2.flatten()[region_list[4]]]
mean_geod_region, mean_l2_region = [],[]
for i in range(len(vis_regions)):
    mean_geod_region.append(np.mean(geod_region[i]))
    mean_l2_region.append(np.mean(l2_region[i]))

fig, ax1 = plt.subplots(dpi=200)
geod_viol = ax1.violinplot(geod_region,showextrema=False)
for pc in geod_viol['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('green')
    pc.set_alpha(0.6)
ax1.plot(np.arange(1,6),mean_geod_region,'go',alpha=0.6)
l2_viol = ax1.violinplot(l2_region,showextrema=False)
for pc in l2_viol['bodies']:
    pc.set_facecolor('red')
    pc.set_edgecolor('red')
    pc.set_alpha(0.6)
ax1.plot(np.arange(1,6),mean_l2_region,'ro',alpha=0.6)
plt.xticks(np.arange(1,6),vis_regions)
plt.ylabel('strength')
plt.savefig('strength_violins.pdf')

#%%
strength = np.load('$PATH$/strength_geod.npy')         
strength2 = np.load('$PATH$/strength_geod2.npy') 
strength_l2 = np.load('$PATH$/strength_l2.npy') 
strength2_l2 = np.load('$PATH$/strength2_l2.npy') 

significance = np.load('$PATH$/significance_geod.npy')         
significance2 = np.load('$PATH$/significance_geod2.npy') 
significance_l2 = np.load('$PATH$/significance_l2_ord.npy') 
significance2_l2 = np.load('$PATH$/significance2_l2.npy') 

     
np.save('$PATH$/strength_geod.npy',strength)         
np.save('$PATH$/strength_geod2.npy',strength2) 
np.save('$PATH$/strength_l2.npy',strength_l2) 
np.save('$PATH$/strength2_l2.npy',strength2_l2) 

np.save('$PATH$/significance_geod.npy',significance)         
np.save('$PATH$/significance_geod2.npy',significance2) 
np.save('$PATH$/significance_l2_ord.npy',significance_l2) 
np.save('$PATH$/significance2_l2.npy',significance2_l2) 

