import Algorithms as alg
import numpy as np
import pickle
import matplotlib.pyplot as plt
#%% Total manifolds
h1_features = np.load('$PATHOF_H1_FEATURES$')
h2_features = np.load('$PATHOF_H2_FEATURES$')
with open('$PATH$\\region_list.pkl', 'rb') as f:
   region_ids = pickle.load(f)
with open('$PATH$\\pers_diags0n.pkl', 'rb') as f:
   pers_diags = pickle.load(f)
data_num = len(h1_features)
vis_regions = ['LP','VISp','VISrl','VISl','VISam']
mfld_labels = ['trivial','circle','sphere','torus','other']
session_labels = np.load('$PATH$\session_labels.npy')
    
ticks = ['(1,0,0)','(1,1,0)','(1,0,1)','(1,2,1)','other']
LP_bars = alg.check_mfld_top(h1_features,h2_features,region_ids[0],compressed=True)
VISp_bars = alg.check_mfld_top(h1_features,h2_features,region_ids[1],compressed=True)
VISrl_bars = alg.check_mfld_top(h1_features,h2_features,region_ids[2],compressed=True)
VISl_bars = alg.check_mfld_top(h1_features,h2_features,region_ids[3],compressed=True)
VISam_bars = alg.check_mfld_top(h1_features,h2_features,region_ids[4],compressed=True)

all_region_bars = np.vstack([LP_bars,VISp_bars,VISrl_bars,VISl_bars,VISam_bars])

LP_adjust = LP_bars/len(region_ids[0])
VISp_adjust = VISp_bars/len(region_ids[1])
VISrl_adjust = VISrl_bars/len(region_ids[2])
VISl_adjust = VISl_bars/len(region_ids[3])
VISam_adjust  = VISam_bars/len(region_ids[4])

all_region_bars_adj = np.vstack([LP_adjust,VISp_adjust,VISrl_adjust,VISl_adjust,VISam_adjust])
#bootstap analysis
all_reg_stat = alg.session_bootstrap_test(h1_features,h2_features,region_ids,session_labels,500)
all_pdns = alg.session_bootstrap_test(h1_features,h2_features,region_ids,session_labels,5000,stat=False)


plt.figure(dpi=200,figsize=(16,1))
plt.bar(ticks,LP_bars,color='mediumblue',width=0.25)
plt.xticks([])
plt.ylim([-1,20])
for i in range(len(ticks)):
    plt.text(ticks[i],3,int(LP_bars[i]), horizontalalignment='center',color='red',size=20)

plt.figure(dpi=200,figsize=(16,1))
plt.bar(ticks,VISp_bars,color='mediumblue',width=0.25)
plt.xticks([])
plt.ylim([-1,20])
for i in range(len(ticks)):
    plt.text(ticks[i],3,int(VISp_bars[i]), horizontalalignment='center',color='red',size=20)

plt.figure(dpi=200,figsize=(16,1))
plt.bar(ticks,VISrl_bars,color='mediumblue',width=0.25)
plt.xticks([])
plt.ylim([-1,20])
for i in range(len(ticks)):
    plt.text(ticks[i],3,int(VISrl_bars[i]), horizontalalignment='center',color='red',size=20)

plt.figure(dpi=200,figsize=(16,1))
plt.bar(ticks,VISl_bars,color='mediumblue',width=0.25)
plt.xticks([])
plt.ylim([-1,20])
for i in range(len(ticks)):
    plt.text(ticks[i],3,int(VISl_bars[i]), horizontalalignment='center',color='red',size=20)

plt.figure(dpi=200,figsize=(16,1))
plt.bar(ticks,VISam_bars,color='mediumblue',width=0.25)
plt.xticks([])
plt.ylim([-1,20])
for i in range(len(ticks)):
    plt.text(ticks[i],3,int(VISam_bars[i]), horizontalalignment='center',color='red',size=20)

#%%manifolds at different frequencies 
temporal_frequencies = [1,2,4,8,15]
orients = [0,45,90,135,180,225,270,315]
top_features = 5
h1_features_fq = []
h2_features_fq = []
LP_bars_fq, VISp_bars_fq, VISrl_bars_fq, VISl_bars_fq, VISam_bars_fq = [],[],[],[],[]
for i in range(len(temporal_frequencies)):
    h1_features_fq.append(np.load('C:\Kosio\Master_Thesis\Data\h1_fdr'+str(i)+'.npy'))
    h2_features_fq.append(np.load('C:\Kosio\Master_Thesis\Data\h2_fdr'+str(i)+'.npy'))
    LP_bars_fq.append(alg.check_mfld_top(h1_features_fq[i],h2_features_fq[i],region_ids[0]))
    VISp_bars_fq.append(alg.check_mfld_top(h1_features_fq[i],h2_features_fq[i],region_ids[1]))
    VISrl_bars_fq.append(alg.check_mfld_top(h1_features_fq[i],h2_features_fq[i],region_ids[2]))
    VISl_bars_fq.append(alg.check_mfld_top(h1_features_fq[i],h2_features_fq[i],region_ids[3]))
    VISam_bars_fq.append(alg.check_mfld_top(h1_features_fq[i],h2_features_fq[i],region_ids[4]))
ticks = ['(1,0,0)','(1,1,0)','(1,0,1)','(1,2,1)','other']
fq_ticks = ['all','1Hz','2Hz','4Hz','8Hz','15Hz']#['all','$0^o$','$45^o$','$90^o$','$135^o$','$180^o$','$225^o$','$270^o$','$315^o$']#

LP_bars_fq = np.asarray(LP_bars_fq)
VISp_bars_fq = np.asarray(VISp_bars_fq)
VISrl_bars_fq = np.asarray(VISrl_bars_fq)
VISl_bars_fq = np.asarray(VISl_bars_fq)
VISam_bars_fq = np.asarray(VISam_bars_fq)

LP_bars_fq_adj = np.asarray(LP_bars_fq)/len(region_ids[0])
VISp_bars_fq_adj = np.asarray(VISp_bars_fq)/len(region_ids[1])
VISrl_bars_fq_adj = np.asarray(VISrl_bars_fq)/len(region_ids[2])
VISl_bars_fq_adj = np.asarray(VISl_bars_fq)/len(region_ids[3])
VISam_bars_fq_adj = np.asarray(VISam_bars_fq)/len(region_ids[4])


#manifold comparison
flat_bars_fq = np.array([LP_bars_fq[:,0],VISp_bars_fq[:,0],VISrl_bars_fq[:,0],VISl_bars_fq[:,0],VISam_bars_fq[:,0]])
circle_bars_fq = np.array([LP_bars_fq[:,1],VISp_bars_fq[:,1],VISrl_bars_fq[:,1],VISl_bars_fq[:,1],VISam_bars_fq[:,1]])
sphere_bars_fq = np.array([LP_bars_fq[:,2],VISp_bars_fq[:,2],VISrl_bars_fq[:,2],VISl_bars_fq[:,2],VISam_bars_fq[:,2]])
torus_bars_fq = np.array([LP_bars_fq[:,3],VISp_bars_fq[:,3],VISrl_bars_fq[:,3],VISl_bars_fq[:,3],VISam_bars_fq[:,3]])
other_bars_fq = np.array([LP_bars_fq[:,4],VISp_bars_fq[:,4],VISrl_bars_fq[:,4],VISl_bars_fq[:,4],VISam_bars_fq[:,4]])

flat_bars_fq_adj = np.array([LP_bars_fq_adj[:,0],VISp_bars_fq_adj[:,0],VISrl_bars_fq_adj[:,0],VISl_bars_fq_adj[:,0],VISam_bars_fq_adj[:,0]])
circle_bars_fq_adj = np.array([LP_bars_fq_adj[:,1],VISp_bars_fq_adj[:,1],VISrl_bars_fq_adj[:,1],VISl_bars_fq_adj[:,1],VISam_bars_fq_adj[:,1]])
sphere_bars_fq_adj = np.array([LP_bars_fq_adj[:,2],VISp_bars_fq_adj[:,2],VISrl_bars_fq_adj[:,2],VISl_bars_fq_adj[:,2],VISam_bars_fq_adj[:,2]])
torus_bars_fq_adj = np.array([LP_bars_fq_adj[:,3],VISp_bars_fq_adj[:,3],VISrl_bars_fq_adj[:,3],VISl_bars_fq_adj[:,3],VISam_bars_fq_adj[:,3]])
other_bars_fq_adj = np.array([LP_bars_fq_adj[:,4],VISp_bars_fq_adj[:,4],VISrl_bars_fq_adj[:,4],VISl_bars_fq_adj[:,4],VISam_bars_fq_adj[:,4]])

flat_bars = np.vstack((all_region_bars[:,0],flat_bars_fq.T)).T
circle_bars = np.vstack((all_region_bars[:,1],circle_bars_fq.T)).T
sphere_bars = np.vstack((all_region_bars[:,2],sphere_bars_fq.T)).T
torus_bars = np.vstack((all_region_bars[:,3],torus_bars_fq.T)).T
other_bars = np.vstack((all_region_bars[:,4],other_bars_fq.T)).T

flat_bars_adj = np.vstack((all_region_bars_adj[:,0],flat_bars_fq_adj.T)).T
circle_bars_adj = np.vstack((all_region_bars_adj[:,1],circle_bars_fq_adj.T)).T
sphere_bars_adj = np.vstack((all_region_bars_adj[:,2],sphere_bars_fq_adj.T)).T
torus_bars_adj = np.vstack((all_region_bars_adj[:,3],torus_bars_fq_adj.T)).T
other_bars_adj = np.vstack((all_region_bars_adj[:,4],other_bars_fq_adj.T)).T

#calculate whether fixed stimulus lead to a different topology structure
flat_reg_diff = mean_difference(flat_bars.T)#
circle_reg_diff = mean_difference(circle_bars.T)#
sphere_reg_diff = mean_difference(sphere_bars.T)#
torus_reg_diff = mean_difference(torus_bars.T)#
other_reg_diff = mean_difference(other_bars.T)#

top_reg_diff = np.vstack((flat_reg_diff,circle_reg_diff,sphere_reg_diff,torus_reg_diff,other_reg_diff))

fig, ax = plt.subplots(1,1,dpi=200)
im1 = ax.imshow(np.expand_dims(top_reg_diff[:,0],0),cmap='coolwarm')
for (j,i),label in np.ndenumerate(np.expand_dims(top_reg_diff[:,0],0)):
    ax.text(i,j,round(label,1),ha='center',va='center',fontsize=16)
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(ticks,fontsize=20)
ax.set_yticks([])

plt_ticks = np.array([0.9,1.9,2.9,3.9,4.9])
bar_colors = ['red','green','orange','purple','blue']
spacing = 0.15

plt.figure(dpi=200,figsize=(10,1.2))
for i in range(len(temporal_frequencies)):
    plt.bar(plt_ticks+i*spacing,LP_bars_fq[i],color=bar_colors[i],width=spacing)

for j in range(len(plt_ticks)):
    for i in range(len(temporal_frequencies)):
        plt.text(plt_ticks[j]+i*spacing,LP_bars_fq[i][j],int(LP_bars_fq[i][j]), horizontalalignment='center',color='black',size=10)
plt.xticks([])
plt.ylim([-1,22])

plt.figure(dpi=200,figsize=(10,1.2))
for i in range(len(temporal_frequencies)):
    plt.bar(plt_ticks+i*spacing,VISp_bars_fq[i],color=bar_colors[i],width=spacing)
for j in range(len(plt_ticks)):
    for i in range(len(temporal_frequencies)):
        plt.text(plt_ticks[j]+i*spacing,VISp_bars_fq[i][j],int(VISp_bars_fq[i][j]), horizontalalignment='center',color='black',size=10)
plt.xticks([])
plt.ylim([-1,22])

plt.figure(dpi=200,figsize=(10,1.2))
for i in range(len(temporal_frequencies)):
    plt.bar(plt_ticks+i*spacing,VISrl_bars_fq[i],color=bar_colors[i],width=spacing)
for j in range(len(plt_ticks)):
    for i in range(len(temporal_frequencies)):
        plt.text(plt_ticks[j]+i*spacing,VISrl_bars_fq[i][j],int(VISrl_bars_fq[i][j]), horizontalalignment='center',color='black',size=10)
plt.xticks([])
plt.ylim([-1,22])

plt.figure(dpi=200,figsize=(10,1.2))
for i in range(len(temporal_frequencies)):
    plt.bar(plt_ticks+i*spacing,VISl_bars_fq[i],color=bar_colors[i],width=spacing)
for j in range(len(plt_ticks)):
    for i in range(len(temporal_frequencies)):
        plt.text(plt_ticks[j]+i*spacing,VISl_bars_fq[i][j],int(VISl_bars_fq[i][j]), horizontalalignment='center',color='black',size=10)
plt.xticks([])
plt.ylim([-1,22])

plt.figure(dpi=200,figsize=(10,1.2))
for i in range(len(temporal_frequencies)):
    plt.bar(plt_ticks+i*spacing,VISam_bars_fq[i],color=bar_colors[i],width=spacing)
for j in range(len(plt_ticks)):
    for i in range(len(temporal_frequencies)):
        plt.text(plt_ticks[j]+i*spacing,VISam_bars_fq[i][j],int(VISam_bars_fq[i][j]), horizontalalignment='center',color='black',size=10)
plt.xticks([])
plt.ylim([-1,22])

