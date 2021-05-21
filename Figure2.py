#%%

#distance error and homology strength as a function of sampling
samples = np.arange(20,1500,50)
nums = len(samples)
frequencies = [2,4,6,8,12,16,32,64]
feature_size = np.zeros(nums)
feature_size_l2 = np.zeros(nums)

dist_geod, dist_l2, hom_geod, hom_l2 = [0]*nums, [0]*nums, [0]*nums, [0]*nums
hom_max_l2, hom_max_geod = [0]*nums, [0]*nums
distance_error = [0]*nums
hom_feature_error = [0]*nums
mean_derror, sem_derror = np.zeros(nums), np.zeros(nums)
pimg_geod, pimg_l2 = [0]*nums, [0]*nums
pimg_error = [0]*nums
mean_perror, sem_perror = np.zeros(nums), np.zeros(nums)
p_imager = pimg.PersistenceImager(pixel_size=0.1,kernel_params={'sigma':np.array([[0.05,0],[0,0.05]])})
p_imager.weight = pimg.weighting_fxns.persistence
p_imager.weight_params = {'n': 2}

for n in range(nums):
    S1 = gen_mfld_period(int(samples[n]),fqs=frequencies).T + (10**-14)*np.random.randn(int(samples[n]),len(frequencies)+2)
    dist_geod[n], hom_geod[n] = full_hom_analysis(S1,metric='geodesic',order=False,R=-1,Eps=0.01,dim=1,perm=None)
    dist_l2[n], hom_l2[n] = full_hom_analysis(S1,metric='LP',order=False,R=-1,Eps=0.01,dim=1,perm=None)
    hom_geod[n], hom_l2[n] = normal_bd_dist(hom_geod[n]), normal_bd_dist(hom_l2[n])
    hom_max_l2[n], hom_max_geod[n] = max_pers(hom_l2[n]), max_pers(hom_geod[n])
    distance_error[n] = np.sqrt((dist_l2[n]-dist_geod[n])**2)#/samples[n]
    mean_derror[n] = np.mean(distance_error[n])
    sem_derror[n] = stats.sem(distance_error[n].flatten())
    hom_feature_error[n] = np.abs(hom_max_l2[n]-hom_max_geod[n])
    pimg_geod[n], pimg_l2[n] = p_imager.transform(hom_geod[n][1]), p_imager.transform(hom_l2[n][1])
    pimg_geod[n], pimg_l2[n] = pimg_geod[n]/np.max(pimg_geod[n]), pimg_l2[n]/np.max(pimg_l2[n])
    pimg_error[n] = np.sqrt((pimg_geod[n]-pimg_l2[n])**2)
    mean_perror[n] = np.mean(pimg_error[n])
    sem_perror[n] = stats.sem(pimg_error[n].flatten())    

#plt.figure(dpi=200)
fig, ax1 = plt.subplots(dpi=200)
plt.grid('on')
ax1.errorbar(samples,mean_derror,yerr=sem_derror,color='k')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Distance difference')
ax2 = ax1.twinx()
ax2.errorbar(samples,mean_perror,yerr=sem_perror,color='r')
ax2.set_ylabel('Persistence image difference',color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.tight_layout()



#%%distance error and Homology as a function of geodesic parameter
samples = 500
frequencies = np.array([2,4,6,8,12,16,32,64])
geod_params = np.logspace(-2,2,50) #50 values
nums = len(geod_params)

hom_geod = [0]*nums
distance_error = [0]*nums
mean_derror, sem_derror = np.zeros(nums), np.zeros(nums)
S1 = gen_mfld_period(samples,fqs=frequencies[:1+nums]).T
d_l2, hom_l2 = full_hom_analysis(S1,metric='LP',order=False,perm=None,dim=1)
hom_l2 = normal_bd_dist(hom_l2)
d_geod, hom_geod = [0]*nums, [0]*nums
pimg_geod, pimg_l2 = [0]*nums, [0]*nums
pimg_error = [0]*nums
mean_perror, sem_perror = np.zeros(nums), np.zeros(nums)
p_imager = pimg.PersistenceImager(pixel_size=0.1,kernel_params={'sigma':np.array([[0.05,0],[0,0.05]])})
p_imager.weight = pimg.weighting_fxns.persistence
p_imager.weight_params = {'n': 2}
pimg_l2 = p_imager.transform(hom_l2[1])
pimg_l2 = pimg_l2/np.max(pimg_l2)
for i in range(nums):
    d_geod[i], hom_geod[i] = full_hom_analysis(S1+(10**-14)*np.random.randn(len(S1),len(S1.T)),metric='geodesic',order=False,perm=None,dim=1,R=-1,Eps=geod_params[i])
    hom_geod[i] = normal_bd_dist(hom_geod[i])
    distance_error[i] = np.sqrt((d_l2-d_geod[i])**2)
    mean_derror[i] = np.mean(distance_error[i])
    sem_derror[i] = stats.sem(distance_error[i].flatten())
    pimg_geod[i]  = p_imager.transform(hom_geod[i][1])
    pimg_geod[i]  = pimg_geod[i]/np.max(pimg_geod[i])
    pimg_error[i] = np.sqrt((pimg_geod[i]-pimg_l2)**2)
    mean_perror[i] = np.mean(pimg_error[i])
    sem_perror[i] = stats.sem(pimg_error[i].flatten())    


fig, ax1 = plt.subplots(dpi=200)
plt.grid('on')
ax1.errorbar(geod_params,mean_derror,yerr=sem_derror,color='k')
ax1.set_xlabel('Neighborhood size parameter')
ax1.set_ylabel('Distance difference')
ax1.set_xscale('log')
ax2 = ax1.twinx()
#ax2.errorbar(geod_params,mean_perror,yerr=sem_perror,color='r')
ax2.plot(geod_params,mean_perror,color='r')
ax2.set_ylabel('Persistence image difference',color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.tight_layout()
#%%distance error and Homology as a function of curvature
samples = 500
frequencies = np.array([2,4,6,8,12,16,32,64])
nums = len(frequencies)
theta = np.linspace(0,2*np.pi,samples,endpoint=False)
curvature = np.zeros([samples,nums])
distance_error = [0]*nums
dmats, homs = [0]*nums, [0]*nums
dmats_l2, homs_l2 = [0]*nums, [0]*nums
mean_derror, sem_derror = np.zeros(nums), np.zeros(nums)
pimg_geod, pimg_l2 = [0]*nums, [0]*nums
pimg_error = [0]*nums
mean_perror, sem_perror = np.zeros(nums), np.zeros(nums)
p_imager = pimg.PersistenceImager(pixel_size=0.1,kernel_params={'sigma':np.array([[0.05,0],[0,0.05]])})
p_imager.weight = pimg.weighting_fxns.persistence
p_imager.weight_params = {'n': 2}

for i in range(nums):
    S1 = gen_mfld_period(samples,fqs=frequencies[:1+i]).T + (10**-14)*np.random.randn(samples,len(frequencies[:1+i])+2)
    curvature[:,i] = curvature_per_point(S1)
    dmats[i], homs[i] = full_hom_analysis(S1,metric='geodesic',order=False,R=-1,Eps=0.01,dim=1,perm=None)
    dmats_l2[i], homs_l2[i] = full_hom_analysis(S1,metric='LP',order=False,R=-1,Eps=0.01,dim=1,perm=None)
    homs[i],homs_l2[i] = normal_bd_dist(homs[i]), normal_bd_dist(homs_l2[i])
    distance_error[i] = np.sqrt((dmats_l2[i]-dmats[i])**2)
    mean_derror[i] = np.mean(distance_error[i])
    sem_derror[i] = stats.sem(distance_error[i].flatten())
    pimg_geod[i], pimg_l2[i] = p_imager.transform(homs[i][1]), p_imager.transform(homs_l2[i][1])
    pimg_geod[i], pimg_l2[i] = pimg_geod[i]/np.max(pimg_geod[i]), pimg_l2[i]/np.max(pimg_l2[i])    
    pimg_error[i] = np.sqrt((pimg_geod[i]-pimg_l2[i])**2)
    mean_perror[i] = np.mean(pimg_error[i])
    sem_perror[i] = stats.sem(pimg_error[i].flatten())    

total_curvature = np.sum(curvature,0)/samples


fig, ax1 = plt.subplots(dpi=200)
plt.grid('on')
ax1.errorbar(total_curvature,mean_derror,yerr=sem_derror,color='k')
ax1.set_xlabel('cuvature')
ax1.set_ylabel('Distance difference')
ax2 = ax1.twinx()
ax2.errorbar(total_curvature,mean_perror,yerr=sem_perror,color='r')
ax2.set_ylabel('Persistence image difference',color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.tight_layout()

#%% distance error and Homology as a function of noise
noise = np.logspace(-15,0,50)
nums = len(noise)
samples = 500
frequencies = np.array([2,4,6,8,16,32,64])
feature_size = np.zeros(nums)
feature_size_l2 = np.zeros(nums)

dist_geod, dist_l2, hom_geod, hom_l2 = [0]*nums, [0]*nums, [0]*nums, [0]*nums
hom_max_l2, hom_max_geod = [0]*nums, [0]*nums
distance_error = [0]*nums
hom_feature_error = [0]*nums
mean_derror, sem_derror = np.zeros(nums), np.zeros(nums)
pimg_geod, pimg_l2 = [0]*nums, [0]*nums
pimg_error = [0]*nums
pimg_error_l2 = [0]*nums
mean_perror, sem_perror = np.zeros(nums), np.zeros(nums)
mean_perror_l2, sem_perror_l2 = np.zeros(nums), np.zeros(nums)
p_imager = pimg.PersistenceImager(pixel_size=0.1,kernel_params={'sigma':np.array([[0.05,0],[0,0.05]])})
p_imager.weight = pimg.weighting_fxns.persistence
p_imager.weight_params = {'n': 2}



for n in range(nums):
    S1 = gen_mfld_period(samples,fqs=frequencies).T + noise[n]*np.random.randn(samples,len(frequencies)+2)
    dist_geod[n], hom_geod[n] = full_hom_analysis(S1,metric='geodesic',order=False,R=-1,Eps=0.1,dim=1,perm=None)
    dist_l2[n], hom_l2[n] = full_hom_analysis(S1,metric='LP',order=False,R=-1,Eps=0.1,dim=1,perm=None)
    hom_geod[n], hom_l2[n] = normal_bd_dist(hom_geod[n]), normal_bd_dist(hom_l2[n])
    hom_max_l2[n], hom_max_geod[n] = max_pers(hom_l2[n]), max_pers(hom_geod[n])
    distance_error[n] = np.sqrt((dist_l2[n]-dist_geod[n])**2)
    mean_derror[n] = np.mean(distance_error[n])
    sem_derror[n] = stats.sem(distance_error[n].flatten())
    hom_feature_error[n] = np.abs(hom_max_l2[n]-hom_max_geod[n])
    pimg_geod[n], pimg_l2[n] = p_imager.transform(hom_geod[n][1]), p_imager.transform(hom_l2[n][1])
    pimg_geod[n], pimg_l2[n] = pimg_geod[n]/np.max(pimg_geod[n]), pimg_l2[n]/np.max(pimg_l2[n])
    pimg_error[n] = np.sqrt((pimg_geod[0]-pimg_geod[n])**2)
    pimg_error_l2[n] = np.sqrt((pimg_geod[0]-pimg_l2[n])**2)
    mean_perror[n] = np.mean(pimg_error[n])
    sem_perror[n] = stats.sem(pimg_error[n].flatten())    
    mean_perror_l2[n] = np.mean(pimg_error_l2[n])
    sem_perror_l2[n] = stats.sem(pimg_error_l2[n].flatten())    



fig, ax1 = plt.subplots(dpi=200)
plt.grid('on')
ax1.errorbar(noise,mean_derror,yerr=sem_derror,color='k')
ax1.set_xlabel('noise')
ax1.set_ylabel('Distance difference')
ax2 = ax1.twinx()
ax1.set_xscale('log')
ax2.errorbar(noise,mean_perror,yerr=sem_perror,color='g')
ax2.errorbar(noise,mean_perror_l2,yerr=sem_perror_l2,color='r')
ax2.set_ylabel('Persistence image difference',color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.tight_layout()


