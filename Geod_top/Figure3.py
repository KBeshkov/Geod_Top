import os
from allensdk.brain_observatory.ecephys.ecephys_session import (
    EcephysSession, 
    removed_unused_stimulus_presentation_columns
)
import Algorithms as alg
import numpy as np


cdir = '$DATAPATH$'
datasets = np.sort(os.listdir(cdir))

alpha=0.05

exp_var = []
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
    num_bins_dg = 2
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

for d in all_data:
    exp_var.append(PCA().fit(d).explained_variance_ratio_)

#%%
mouse_num = 51

session_id = datasets[12]
nwb_path = '$DATAPATH$' + str(session_id)
#
session = EcephysSession.from_nwb_path(nwb_path, api_kwargs={
    "amplitude_cutoff_maximum": np.inf,
    "presence_ratio_minimum": -np.inf,
    "isi_violations_maximum": np.inf
})

num_bins_dg = 2#
stim_dat = open_data(session,'drifting_gratings','VISp',num_bins_dg,bin_strt=0,frate=1,snr=2)
pcloud_dg = np.squeeze(np.vstack(stim_dat[0]))

LP_ = full_hom_analysis(pcloud_dg,metric='LP',order=False,R=-1,Eps=.1)
geod_ = full_hom_analysis(pcloud_dg,metric='geodesic',order=False,R=-1,Eps=.1)
LP_hom = normal_bd_dist(LP_[1])
geod_hom = normal_bd_dist(geod_[1])

plt.figure()
plot_diagrams(geod_hom,size=60)

pca_mfld = PCA().fit_transform(pcloud_dg)

fig=plt.figure(dpi=200)
plt.plot(pca_mfld[:,0],pca_mfld[:,1],' c.')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.text(np.max(pca_mfld[:,0])/1.65,np.max(pca_mfld[:,1])-3,str(session_id[8:-4]),fontsize=14)
plt.text(np.max(pca_mfld[:,0])/1.65,np.max(pca_mfld[:,1])-13,'VISp',fontsize=14)

plt.figure(dpi=200)
for i in range(len(exp_var)):
    plt.plot(exp_var[i][:9],'r',alpha=0.1)
plt.plot(exp_var[rat_num][:9],'k-o')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.figure(dpi=200)
plt.subplot(2,2,1)
plt.imshow(LP_[0])
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(geod_[0])
plt.axis('off')
plt.subplot(2,1,2)
plt.hist(np.triu(geod_[0])[np.triu(geod_[0])!=0],100,color='green')
plt.hist(np.triu(LP_[0])[np.triu(LP_[0])!=0],100,color='brown')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(['Geodesic distance','Euclidean distance'])
plt.tight_layout()

plt.figure(dpi=200)
plot_diagrams(LP_hom,size=40,legend=False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.figure(dpi=200)
plot_diagrams(geod_hom,size=40,legend=False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
