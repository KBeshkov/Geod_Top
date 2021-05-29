'''Requires that Figure5_6.py is ran before it, so that the necessary variables are defined'''
import Algorithms as alg
import numpy as np
#%% Statistical tests
all_adj = [all_region_bars_adj,LP_bars_fq_adj,VISp_bars_fq_adj,VISrl_bars_fq_adj,VISl_bars_fq_adj,VISam_bars_fq_adj]

#per region/condition test
sign_test = np.zeros([6,5,5])
sign_fq_test = np.zeros([5,25])
for i in range(len(all_adj)):
    sign_test[i,:,:] = alg.percent_of_score_test(all_pdns,all_adj[i])
alg.stat_image(sign_test[0],mfld_labels,vis_regions)

sign_fq_test = np.vstack(sign_test[1:,:,:])
pmask = np.where(sign_fq_test>0.5)
sign_fq_test[pmask[0],pmask[1]] = 1-sign_fq_test[pmask[0],pmask[1]]
sign_fq_test = np.reshape(fdrcorrection(sign_fq_test.flatten())[1],(25,5))
sign_fq_test[pmask[0],pmask[1]] = 1-sign_fq_test[pmask[0],pmask[1]]

fq_ticks_ = []
for i in vis_regions:
    for j in fq_ticks[1:]:
        fq_ticks_.append(i+' - '+j)

fig, ax = plt.subplots(1,1,dpi=200)
im1 = ax.imshow(sign_fq_test,cmap='coolwarm',aspect='auto')
for (j,i),label in np.ndenumerate(sign_fq_test):
    if label<=0.025 or label>=0.975:
        ax.text(i,j,'*',ha='center',va='center',fontsize=6)
    else:
        ax.text(i,j,round(label,3),ha='center',va='center',fontsize=6)
ax.set_xticks(np.arange(0,5))
ax.set_yticks(np.arange(0,25))
ax.set_xticklabels(mfld_labels,fontsize=12)
ax.set_yticklabels(fq_ticks_,fontsize=6)

#%% Effect sizes

all_fq = alg.session_bootstrap_test(h1_features,h2_features,region_ids,session_labels,5000,perm=False,stat=False)
Hz1_fq = alg.session_bootstrap_test(h1_features_fq[0],h2_features_fq[0],region_ids,session_labels,1000,perm=False,stat=False)
Hz2_fq = alg.session_bootstrap_test(h1_features_fq[1],h2_features_fq[1],region_ids,session_labels,1000,perm=False,stat=False)
Hz4_fq = alg.session_bootstrap_test(h1_features_fq[2],h2_features_fq[2],region_ids,session_labels,1000,perm=False,stat=False)
Hz8_fq = alg.session_bootstrap_test(h1_features_fq[3],h2_features_fq[3],region_ids,session_labels,1000,perm=False,stat=False)
Hz15_fq = alg.session_bootstrap_test(h1_features_fq[4],h2_features_fq[4],region_ids,session_labels,1000,perm=False,stat=False)
Hz_fq = [Hz1_fq,Hz2_fq,Hz4_fq,Hz8_fq,Hz15_fq]

all_eff_size = np.zeros([5,5,5])
fq_eff_sizes = np.zeros([5,5,5])
for i in range(len(all_fq)):
    for j in range(len(all_fq)):
        for k in range(len(all_fq)):
            fq_eff_sizes[i,j,k] =cohen_d(Hz_fq[k][i,j,:],all_fq[i,j,:])
            if i>j:
                all_eff_size[i,j,k] = np.abs(cohen_d(all_fq[i,k,:].flatten(),all_fq[j,k,:].flatten()))   

fq_eff_sizes = np.concatenate(fq_eff_sizes,1).T
fq_eff_sizes[np.isnan(fq_eff_sizes)] = 0

fig, ax = plt.subplots(1,1,dpi=200)
im1 = ax.imshow(fq_eff_sizes,cmap='coolwarm',aspect='auto')
for (j,i),label in np.ndenumerate(fq_eff_sizes):
    ax.text(i,j,round(label,3),ha='center',va='center',fontsize=6)
ax.set_xticks(np.arange(0,5))
ax.set_yticks(np.arange(0,25))
ax.set_xticklabels(mfld_labels,fontsize=12)
ax.set_yticklabels(fq_ticks_,fontsize=6)
