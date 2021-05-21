import Algorithms as alg
import numpy as np
from scipy.stats import percentileofscore
import pickle
import os
from joblib import Parallel, delayed
from ripser import ripser as tda
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession



cdir = '$PATHOFDATA$'
datasets = np.sort(os.listdir(cdir))

alpha=0.05/126 #
m_tst = 3 #maximum number of features to look for
stim_type = 'drifting_gratings'
stim_num = 'all'

pdiags = []
pdiags_l2 = []

exp_var = []
dmat_geod = []
dmat_l2 = []
count =  0
all_data = []
region_list = [[]]*5
stims = [[],[],[],[],[]]#,[],[],[],[]]
session_labels = []
session_num = 0
for session_id in datasets:
    nwb_path = cdir + str(session_id)
#
    session = EcephysSession.from_nwb_path(nwb_path, api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf
    })
    print(session_id)
    temp_stim = session.stimulus_presentations[session.stimulus_presentations['stimulus_name'] == stim_type]
    inv_trials = alg.invalid_trials_ind(session,'drifting_gratings')
    stims[0].append(np.where(np.logical_and(temp_stim.temporal_frequency==1,inv_trials!=1)))
    stims[1].append(np.where(np.logical_and(temp_stim.temporal_frequency==2,inv_trials!=1)))
    stims[2].append(np.where(np.logical_and(temp_stim.temporal_frequency==4,inv_trials!=1)))
    stims[3].append(np.where(np.logical_and(temp_stim.temporal_frequency==8,inv_trials!=1)))
    stims[4].append(np.where(np.logical_and(temp_stim.temporal_frequency==15,inv_trials!=1)))
    vis_regions = ['LP','VISp','VISrl','VISl','VISam']
    stim_dat = {}
    trial_dat = {}
    pcloud_dg = {}
    num_bins_dg = 2 #bin length
    for i in vis_regions: #set inval to False when analyzing frequency fixed stimuli
        temp_data = alg.open_data(session,'drifting_gratings',i,num_bins_dg,bin_strt=0,frate=1,snr=2,inval=True)
        stim_dat[str(i)+' dg'] = temp_data[0]
        pcloud_dg[str(i)] = np.vstack([temp_data[0]])
    region_ids = []
    count_region = 0
    pcloud_mean_dg = {}
    for i in vis_regions:
        pcloud_mean_dg[str(i)] = []
        for j in range(len(stim_dat[str(i)+' dg'])):
                pcloud_mean_dg[str(i)].append(stim_dat[str(i)+' dg'][j])#[:,region_tun_curves[count]]
        if len(pcloud_mean_dg[str(i)])!=0:
            pcloud_mean_dg[str(i)] = np.squeeze(np.vstack(pcloud_mean_dg[str(i)]))
            if np.shape(np.squeeze(pcloud_mean_dg[str(i)]))[1]>5:
                print(np.shape(np.squeeze(pcloud_mean_dg[str(i)]))[1])
                region_ids.append(count_region)
                region_list[count_region] = region_list[count_region]+[len(all_data)]
                #run for specific stiulus
#                all_data.append(np.squeeze(pcloud_mean_dg[str(i)][stims[stim_num][session_num],:]))
                #run for all stimuli
                all_data.append(np.squeeze(pcloud_mean_dg[str(i)]))#[stims[stim_num][session_num],:]))
                session_labels.append(session_num)
            else:
                print('removed '+str(i))
                del pcloud_mean_dg[str(i)]
            count_region = count_region+1
    vis_regions = list(pcloud_mean_dg.keys())
    session_num = session_num+1

np.save('session_labels.npy',np.asarray(session_labels))

significance = np.ones(len(all_data))#np.ones([5,len(datasets)])
significance2 = np.ones(len(all_data))#np.ones([5,len(datasets)])


geod_params = [0.1] #geodesic parameter
percentile_geod = np.zeros(len(all_data))
percentile_geod2 = np.zeros(len(all_data))

pvals_h1 = []
pvals_h2 = []

h1_betti = np.zeros(len(all_data))
h2_betti = np.zeros(len(all_data))
for g in geod_params:
    count=0
    n_permutations = 2500
    max_hom_dg1 = np.zeros([n_permutations,len(all_data)])
    max_hom_dg1_l2 = np.zeros([n_permutations,len(all_data)])
    max_hom_dg2 = np.zeros([n_permutations,len(all_data)])
    max_hom_dg2_l2 = np.zeros([n_permutations,len(all_data)])
    for d in all_data:  
#        d = alg.exclude_neighbors(d,perc=2.5,pr=25)
        #permutation of mean trajectories for persistence significance threshold
        dmat_temp = alg.geodesic(d,eps=g)
        dmat_geod.append(dmat_temp)
        hom_dg_temp = alg.normal_bd_dist(tda(dmat_temp,distance_matrix=True,maxdim=2,n_perm=None)['dgms'])
        pdiags.append(hom_dg_temp)
        n_nrns = len(d.T)
        n_times = len(d)
        perm_analysis = np.asanyarray(Parallel(n_jobs=-1,backend='loky')(delayed(alg.full_perm_analysis)(d,geod_ep=g) for p in range(n_permutations)))
        max_hom_dg1[:,count] = perm_analysis[:,0]
        max_hom_dg2[:,count] = perm_analysis[:,1]        
        if len(hom_dg_temp[2])>=3:
            h2_pers = np.sort(pdiags[count][2][:,1]-pdiags[count][2][:,0])
            h2_pvals = np.zeros(m_tst)
            for m in range(m_tst):
                h2_pvals[m] = 1-percentileofscore(max_hom_dg2[:,count],h2_pers[-m-1])/100
                pvals_h2.append(h2_pvals[m])
            h2_pvals = alg.fdr(h2_pvals)
            h2_betti[count] = np.sum(h2_pvals<=alpha)
        else:
            for m in range(m_tst):
                pvals_h2.append(1)
            h2_betti[count] = 0
        h1_pers = np.sort(pdiags[count][1][:,1]-pdiags[count][1][:,0])
        h1_pvals = np.zeros(m_tst)
        for m in range(m_tst):
            h1_pvals[m] = 1-percentileofscore(max_hom_dg1[:,count],h1_pers[-m-1])/100
            pvals_h1.append(h1_pvals[m])
        h1_pvals = alg.fdr(h1_pvals)
        h1_betti[count] = np.sum(h1_pvals<=alpha)
        count = count + 1
        print(count)


pvals_h1 = alg.fdr(np.asarray(pvals_h1))
pvals_h2 = alg.fdr(np.asarray(pvals_h2))

np.save('$PATH$/h1_pvals'+str(stim_num)+'.npy', pvals_h1)
np.save('$PATH$/h2_pvals'+str(stim_num)+'.npy', pvals_h2)
np.save('$PATH$/h1_betti'+str(stim_num)+'.npy', h1_betti)
np.save('$PATH$/h2_betti'+str(stim_num)+'.npy',h2_betti)
with open('$PATH$/pers_diags'+str(stim_num)+'.pkl', 'wb') as f:
    pickle.dump(pdiags, f)
with open('$PATH$/region_list.pkl', 'wb') as h:
    pickle.dump(region_list, h)