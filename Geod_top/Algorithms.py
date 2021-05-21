#implementation of algorithms
from sklearn.manifold import Isomap
import numpy as np
import numpy.matlib as mat
import math
import numpy.matlib
from scipy import stats
from scipy.stats import rankdata
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
import PersistenceImages.persistence_images as pimg
from sklearn.metrics import pairwise_distances
from ripser import ripser as tda
from persim import plot_diagrams
import umap.umap_ as umap
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.gridspec as gridspec
import time as time
from scipy.stats import percentileofscore
from scipy import linalg
from scipy.special import legendre, sph_harm, eval_legendre
from statsmodels.stats.multitest import fdrcorrection
import pickle


def invalid_trials_ind(sess,stim_condition):
    stim = sess.stimulus_presentations[sess.stimulus_presentations['stimulus_name'] == stim_condition].index.values
    q = []
    if len(sess.invalid_times)>0:
        iv_times_start = sess.invalid_times.start_time.values
        iv_times_stop = sess.invalid_times.stop_time.values
        o = sess.stimulus_presentations
        s_times = o.start_time[stim].values
        for i in range(len(iv_times_start)):
            q.append(np.logical_and(iv_times_start[i]<=s_times,s_times<=iv_times_stop[i]))
    return sum(q)

def open_data(sess,stim_condition,region,bin_step,bin_strt=0,snr=2,frate=1,inval=True):
    stim = sess.stimulus_presentations[sess.stimulus_presentations['stimulus_name'] == stim_condition].index.values
#    stim = sess.stimulus_presentations[np.logical_and(
#        sess.stimulus_presentations['stimulus_name'] == stim_condition,sess.stimulus_presentations[
#                'phase']=='0.0')].index.values
    decent_snr_unit_ids = sess.units[np.logical_and(sess.units['snr'] >= snr, sess.units['firing_rate'] > frate)]        
    decent_snr_unit_ids = list(decent_snr_unit_ids[decent_snr_unit_ids['ecephys_structure_acronym']==region].index.values)
    if inval==True:
        if len(sess.invalid_times)>0:
            iv_times_start = sess.invalid_times.start_time.values
            iv_times_stop = sess.invalid_times.stop_time.values
            o = sess.stimulus_presentations
            s_times = o.start_time[stim].values
            q = []
            for i in range(len(iv_times_start)):
                q.append(np.logical_and(iv_times_start[i]<=s_times,s_times<=iv_times_stop[i]))
            stim = stim[np.where(sum(q)!=1)]
    durations = sess.stimulus_presentations.duration[stim]
    bins = []
    for i in range(len(durations)):
        bins.append(np.arange(bin_strt,np.max(durations.values),bin_step))
    stim_labels = list(sess.stimulus_presentations.stimulus_condition_id[stim])
    spikes_per_stim = []
    count = 0
    for i in np.unique(stim_labels):
        spike_counts_da = sess.presentationwise_spike_counts(
            bin_edges=bins[count],
            stimulus_presentation_ids=stim[stim_labels==i],
            unit_ids=decent_snr_unit_ids,
            binarize=False
        )
        
        spikes_per_stim.append(spike_counts_da.data)
        count = count + 1
    return [spikes_per_stim,durations]
    
def mean_difference(X):
    N = len(X[:,0])
    D = np.zeros([N])
    for i in range(N):
        v = np.sum(X[i,:])
        for j in range(N):
            D[i] = D[i] - (v - np.sum(X[j,:]))
    return D/(N-1)
    
'''
    Normalized birth/death distance:
    Outputs the persistence of a feature normalized between 0 and 1. In other words
    (death_of_feature-birth_of_feature)/upper_bound

    Input: a birth/death diagram
    
    Output: a birth/death diagram with normalized distances
'''

def normal_bd_dist(x):
    x_copy = np.copy(x)
    a = np.concatenate(x_copy).flatten()
    finite_dgm = a[np.isfinite(a)]
    ax_min, ax_max = np.min(finite_dgm), np.max(finite_dgm)
    x_r = ax_max - ax_min

    buffer = x_r / 5

    x_down = ax_min - buffer / 2
    x_up = ax_max + buffer

    y_down, y_up = x_down, x_up
    yr = y_up - y_down
    b_inf = y_down + yr * 0.95
    norm_pers = []
    for i in range(len(x)):
        norm_pers.append((x[i])/b_inf)
    return norm_pers    
 

def connectedness(X):
    d = pairwise_distances(X)
    hom = tda(d,distance_matrix=True,maxdim=0)['dgms'][0]
    return hom[-2,1]

def count_neighbors(D,rad='auto'):
    if rad == 'auto':
        rad = np.percentile(D.flatten(),5)
        Nbrs = np.sum(D<rad,0)-1
    else:
        Nbrs = np.sum(D<rad,0)-1
    return Nbrs
    
def exclude_neighbors(X,perc,pr=25):
    d = pairwise_distances(X)
    nbrs = count_neighbors(D=d,rad='auto',p=pr)
    Xn = X[nbrs>np.percentile(nbrs,perc),:]  
    return Xn

'''Numeric calcultion of geodesics using a shortest path graph algorithm'''
def geodesic(X,r=-1,eps=0.1,count=1):
    Xn = np.copy(X)
    if r>0:
        N = len(Xn)
        d = pairwise_distances(Xn)
        d_geod = (10**10)*np.ones([N,N])
        neighbors = []
        for i in range(N):
            neighbors=np.where(d[i,:]<r)[0]
            d_geod[i,neighbors] = d[i,neighbors]
        d_geod = shortest_path(d_geod)
        if np.sum(d_geod>=10**10)>0:
            count += 1
            dn = geodesic(Xn,r=r+eps,eps=eps,count=count)
            return dn
        else:
            print('finished in ' + str(count) + ' recursions')
            return d_geod
    else:
        N = len(Xn)
        d = pairwise_distances(Xn)
        hom = tda(d,distance_matrix=True,maxdim=0)['dgms'][0]
        r = hom[-2,1]+eps*hom[-2,1]
        d_geod = (10**10)*np.ones([N,N])
        neighbors = []
        for i in range(N):
            neighbors=np.where(d[i,:]<r)[0]
            d_geod[i,neighbors] = d[i,neighbors]
        d_geod = shortest_path(d_geod)
        if np.sum(d_geod>=10**10)>0:
            count += 1
            dn = geodesic(Xn,r=r+0.1*r,eps=0.1*r,count=count)
            return dn
        else:
            print('finished in ' + str(count) + ' recursions')

        return d_geod


def norm_mat(X):
    mat_max = np.max(X)
    mat_min = np.min(X)
    X = (X-mat_min)/mat_max
    return(X)
    
    
def order_complex(D):
    ord_mat = np.triu(D)
    np.fill_diagonal(ord_mat,0)
    Ord = rankdata(ord_mat.flatten(),method='dense').reshape(np.shape(D))
#    inv_ranks = np.sum(Ord==1)
    Ord = np.triu(Ord)+np.triu(Ord).T
    Ord = Ord #- inv_ranks
    np.fill_diagonal(Ord,0)
    return Ord/np.max(Ord)

    
    
def fake_data(N,T,frate,bins):
    X = np.zeros([bins,N,T])
    for b in range(bins):
        for i in range(N):
            rand_v = np.sort(np.random.choice(np.linspace(0,T-1,T),size=frate[i],replace=False)).astype(int)
            X[b,i,rand_v] = 1
    return X


''' Permutation test using the Monte-Carlo method based on the sum of L2 distances between two 
    sets of multidimensional vectors in R^n'''

def mc_perm_test(X,Y,nperms):
    xlen,ylen = len(X),len(Y)
    D_true = np.zeros([xlen,ylen])
    for i in range(xlen): #compute the true distance based test statistic
        for j in range(ylen):
            D_true[i,j] = np.sqrt(np.sum(X[i]-Y[j])**2)
    D_stat = np.sum(D_true)
    Z = np.concatenate([X,Y])
    p_val = 0
    for k in range(nperms): #permute the entries of the concatenated array
        perm_idx = np.random.permutation(xlen+ylen)
        Z = Z[perm_idx,:]
        D_shuff = np.zeros([xlen,ylen])
        for i in range(xlen):
            for j in range(ylen):
                D_shuff[i,j] = np.sqrt(np.sum(Z[i,:]-Z[xlen+j,:])**2)
        if np.sum(D_shuff)>D_stat:
            p_val = p_val + 1
    return [D_stat,p_val/nperms]
    

'''subsample array'''
def subsample(X,percent=0.05):
    resamp_X = np.copy(X)
    sub_idx = np.random.choice(np.arange(0,len(X)),int(len(X)*percent))
    resample = np.random.choice(np.arange(0,len(X)),int(len(X)*percent)).astype(int)
    resamp_X[sub_idx] = resamp_X[resample]
    return resamp_X

    
def subsamp_remove(X,n_removed):
    subsamp_X = np.random.permutation(X)[:-n_removed]
    return subsamp_X

''' Bootstrapping the mean with replacement for estimating confidence intervals '''

def bootstrap_ci(X,nsamp):
    param_pdf = np.zeros(nsamp)
    dat_param = np.mean(X)
    for i in range(nsamp):
        resample = np.random.choice(np.arange(0,len(X)),len(X)).astype(int)
        param_pdf[i] = np.mean(X[resample])
    ci1 = np.percentile(param_pdf,5)
    ci2 = np.percentile(param_pdf,95)
    return [dat_param,(ci1,ci2)]
        
def permute_mult(X):
    Y = np.zeros([len(X),len(X[0,:])])
    for i in range(len(X)):
        Y[i,:] = X[i,np.random.permutation(len(X[0,:]))]
    for i in range(len(X[0,:])):
        Y[:,i] = Y[np.random.permutation(len(X)),i]
    return Y

def perm_test_multdim(X,nperm,mean_d=False):
    N = len(X)
    M = len(X.T)
    if mean_d==False:
        real_dist = pairwise_distances(X)
        d_distrib = np.zeros([N,N,nperm])
        for i in range(nperm):
            Y = np.zeros([N,M])
            for j in range(M):
                Y[:,j] = X[np.random.permutation(N),j]
            d_distrib[:,:,i] = pairwise_distances(Y)
        sign_val = np.zeros([N,N])
        for i in range(N):
            for j in range(N):
                if i>j:
                    sign_val[i,j] = (100-percentileofscore(d_distrib[i,j,:].flatten(),real_dist[i,j]))/100
        return sign_val, d_distrib
    else:
        real_dist = np.sum(pairwise_distances(X),0)
        d_distrib = np.zeros([N,nperm])
        for i in range(nperm):
            Y = np.zeros([N,M])
            for j in range(M):
                Y[:,j] = X[np.random.permutation(N),j]
            d_distrib[:,i] = np.sum(pairwise_distances(Y),0)
        sign_val = np.zeros(N)
        for i in range(N):
            sign_val[i] = (100-percentileofscore(d_distrib[i,:].flatten(),real_dist[i]))/100
        return sign_val, d_distrib


def fnct(x):
    return x + 0.5*np.sin(2*x)

def f_inv(x,er = 10**-8):
    xn = x + 0.0001*np.random.randn()
    y = fnct(xn)
    while abs(y-x)>er:
        xn -= (y-x)/(1+np.cos(2*xn))
        y = fnct(xn)
    return xn


'''Finds the most persistent component of a given dimension > 0'''
def max_pers(pd,dim=1):
    if len(pd[dim])>0:
        pers = pd[dim][:,1] - pd[dim][:,0]
        max_persistence = np.max(pers)
        return max_persistence
    else:
        return 0
    

    
def full_hom_analysis(X,metric='LP',order=True,q=2,pimage=False,dim=2,perm=200,R=0.1,Eps=0.1):
    if metric == 'LP':
        if order==True:
            dmat = order_complex(pairwise_distances(X))
        else:
            dmat = pairwise_distances(X)
    elif metric == 'geodesic':
        if order==True:
            dmat = order_complex(geodesic(X,r=R,eps=Eps))
        else:
            dmat = geodesic(X,r=R,eps=Eps)
    hom = tda(dmat,distance_matrix=True,maxdim=dim,n_perm=perm)['dgms']
    return [dmat,hom]
      

def spherical_harmonics(l,m,N,R=1):
    t1 = np.linspace(0,2*np.pi,N)
    t2 = np.linspace(0,np.pi,N)
    theta,phi = np.meshgrid(t1,t2)
    theta, phi = theta.flatten(), phi.flatten()
    x = R*np.cos(theta)*np.sin(phi)
    y = R*np.sin(theta)*np.sin(phi)
    z = R*np.cos(phi)
    Y = sph_harm(m,l,theta,phi)
    if m < 0:
        Y = Y.imag
    elif m >= 0:
        Y = Y.real
    Yx, Yy, Yz = np.abs(Y).T * np.array([x,y,z])
    return Y,np.vstack([Yx,Yy,Yz]).T 
        
def gen_mfld_period(N,R=1,fqs=1,mtype='S1'):
    t = np.linspace(0,2*np.pi,N,endpoint=False)
    if mtype=='S1':
        x = R*np.cos(t)
        y = R*np.sin(t)
        z = []
        for i in range(len(fqs)):
            z.append(R*np.sin(fqs[i]*t))
        return np.vstack((x,y,z))
    elif mtype=='S2':
        t_ = np.linspace(0,np.pi,N)
        t1,t2 = np.meshgrid(t,t_)
        t1,t2 = t1.flatten(),t2.flatten()
        x = R*np.cos(t1)*np.sin(t2)
        y = R*np.sin(t1)*np.sin(t2)
        z = R*np.cos(t2)
        u = []
        for i in range(len(fqs)):
            u.append(spherical_harmonics(fqs[i],0,N,R)[0]*np.sin(fqs[i]*t1))#R*np.cos(fqs[i]*t2)*np.sin(fqs[i]*t1))#R*np.cos(fqs[i]*t2)*np.sin(fqs[i]*t2))##
        return np.vstack((x,y,z,u))
    elif mtype=='sph_harm':
        t_ = np.linspace(0,np.pi,N)
        t1,t2 = np.meshgrid(t,t_)
        t1,t2 = t1.flatten(),t2.flatten()
        
        u = []
        inv_sin = np.zeros(len(t1))
        for i in range(len(t1)):
            if np.sin(t1[i])>0:
                inv_sin[i] = np.sqrt(np.sin(t1[i]))
            if np.sin(t1[i])<0:
                inv_sin[i] = np.sqrt(-np.sin(t1[i]))
        for i in range(len(fqs)):
            u.append(spherical_harmonics(fqs[i],0,N,R)[0]*np.sin(t1))#R*np.cos(fqs[i]*t2)*np.sin(fqs[i]*t2))##
        return np.vstack(u)
    elif mtype=='legendre_functs':
        t = np.linspace(-1,1,N)
        u = []
        for i in range(len(fqs)):
            u.append(eval_legendre(fqs[i],t)/np.sum(np.abs(eval_legendre(fqs[i],t))))
        return np.vstack(u)
        

def mfld_derivs(fqs,angle):
    dx = np.zeros([len(fqs),len(angle)])
    dx2 = np.zeros([len(fqs),len(angle)])
    for i in range(len(fqs)):
        dx[i,:] = fqs[i]*np.cos(fqs[i]*angle)
        dx2[i,:] = (fqs[i]**2)*np.sin((fqs[i]**2)*angle)
    return dx,dx2

    
def elipse_2nd_der(theta,a=0):
    return np.squeeze(np.array([[-np.cos(theta)-a*np.sin(theta)],[-np.sin(theta)-a*np.cos(theta)]]))
    

def curvature_per_point(M):
    dims  = len(M.T)
    dx_dt = np.zeros(np.shape(M))
    for i in range(dims):      
        dx_dt[:,i] = np.gradient(M[:,i])
    ds_dt = np.sqrt(np.sum(np.multiply(dx_dt,dx_dt),1))
    tangent = np.array([1/ds_dt]*dims).T*dx_dt
    d_tang = np.zeros(np.shape(M))
    for i in range(dims):
        d_tang[:,i] = np.gradient(tangent[:,i])
    curvature = np.sqrt(np.sum(np.multiply(d_tang,d_tang),1))
    return curvature
    

def check_mfld_top(X1,X2,reg_ids,compressed=True):
    if compressed == True:
        X_bars = np.zeros(5)
        X_bars[0] = np.sum(np.logical_and(X1[reg_ids]==0,X2[reg_ids]==0))
        X_bars[1] = np.sum(np.logical_and(X1[reg_ids]==1,X2[reg_ids]==0))
        X_bars[2] = np.sum(np.logical_and(X1[reg_ids]==0,X2[reg_ids]==1))
        X_bars[3] = np.sum(np.logical_and(X1[reg_ids]==2,X2[reg_ids]==1))
        X_bars[4] = len(reg_ids)-np.sum(X_bars[:4])
        return X_bars
    else:
        X_bars = np.zeros(16)
        count = 0
        for i in range(4):
            for j in range(4):
                X_bars[count] = np.sum(np.logical_and(X1[reg_ids]==j,X2[reg_ids]==i))
                count = count + 1
        return X_bars
        
def full_perm_analysis(dat,dim=2,geod_ep=0.2,metric= 'geod'):
    x_dg_temp = permute_mult(dat)
    if metric=='geod':   
        dmat_temp = geodesic(x_dg_temp,eps=geod_ep)
    elif metric=='LP':
        dmat_temp = pairwise_distances(x_dg_temp)
    hom_dg_temp = normal_bd_dist(tda(dmat_temp,distance_matrix=True,maxdim=dim,n_perm=None)['dgms'])
    hom1 = max_pers(hom_dg_temp,dim=1)
    if dim==2:
        hom2 = max_pers(hom_dg_temp,dim=2)
        return [hom1,hom2]
    else:
        return hom1        
    
def fdr(p_vals):
    return fdrcorrection(p_vals)[1]

def mfld_bootstrap(h1,h2,region_ids,n_sub,n_perm):
     X = np.zeros([5,5,n_perm])
     for i in range(n_perm):
         rand_sub = np.sort(np.random.choice(len(h1),len(h1)-n_sub,replace=False))
         h1_perm = h1[rand_sub]
         h2_perm = h2[rand_sub]
         region_ids_perm = []
         for j in region_ids:
             reg_permlist = []
             intersect = np.intersect1d(j,rand_sub)
             for r in range(len(intersect)):
                 reg_permlist.append(np.where(intersect[r]==rand_sub)[0][0])
             region_ids_perm.append(reg_permlist)
         LP_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[0])
         VISp_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[1])
         VISrl_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[2])
         VISl_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[3])
         VISam_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[4])
         X[:,:,i] = np.vstack([LP_bars,VISp_bars,VISrl_bars,VISl_bars,VISam_bars])
     return X
    
    
    
def model_distrib_comp(pd_bar,region_ids,nsamp,nperm,N=5):
    pdns = np.zeros([np.shape(pd_bar)[0],np.shape(pd_bar)[1],nperm])
    pvals_all = np.zeros([np.shape(pd_bar)[0],np.shape(pd_bar)[1],nperm])
    for p in range(nperm):
        for n in range(len(pd_bar)):
            temp_dns = np.random.choice(np.arange(0,N),size=nsamp,p=pd_bar[n])
            for i in range(N):
                pdns[n,i,p] = np.sum(temp_dns==i)
        LP_bars = pdns[0,:,p]
        VISp_bars = pdns[1,:,p]
        VISrl_bars = pdns[2,:,p]
        VISl_bars = pdns[3,:,p]
        VISam_bars = pdns[4,:,p]
        X = np.vstack([LP_bars,VISp_bars,VISrl_bars,VISl_bars,VISam_bars])
        pvals_all[:,:,p] = perm_test_multdim(X,1000)[0]
    return X,pvals_all


def session_bootstrap_test(h1,h2,region_ids,sessions,nperm,perm=True,stat=True):
    n_regions = len(region_ids)
    X = np.zeros([n_regions,5,nperm])
    X_adj = np.zeros([n_regions,5,nperm])
    un_sess = np.unique(sessions)
    nsamp = len(sessions)
    for i in range(nperm):
        rand_sub = np.sort(np.random.choice(un_sess,nsamp))
        h1_temp = []
        h2_temp = []
        temp_regions = [[],[],[],[],[]]
        count = 0
        for r in rand_sub:
            h1_temp.append(h1[sessions==r])
            h2_temp.append(h2[sessions==r])
            sess_reg = np.where(sessions==r)[0]
            for n in range(n_regions):
                if np.sum(np.in1d(sess_reg,region_ids[n]))!=0:
                    temp_regions[n].append(count)
                    count = count+1
        h1_temp = np.hstack(h1_temp)
        h2_temp = np.hstack(h2_temp)
        X_bars = []
        X_bars_adj = []
        for j in range(n_regions):
            X_bars.append(check_mfld_top(h1_temp,h2_temp,temp_regions[j]))
            X_bars_adj.append(check_mfld_top(h1_temp,h2_temp,temp_regions[j])/len(temp_regions[j]))
        X[:,:,i] = np.vstack(X_bars)
        if perm == True:
            xadj = np.vstack(X_bars_adj)
            np.random.shuffle(xadj)
            X_adj[:,:,i] = xadj
        else:
            X_adj[:,:,i] = np.vstack(X_bars_adj)
    if stat==True:
        pvals = np.zeros([5,5])
        for i in range(5):
            for j in range(5):
                if i>j:
                    pvals[i,j] = stats.wilcoxon(X_adj[i,:,:].flatten(),X_adj[j,:,:].flatten())[1]
        return X,pvals,X_adj
    return X_adj
    
    
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2)/dof)


def percent_of_score_test(X,Y):
    pvals = np.zeros([np.shape(X)[0],np.shape(X)[1]])
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            pvals[i,j] = percentileofscore(X[:,j,:].flatten(),Y[i,j])/100
    return pvals

def stat_image(X,xtick,ytick,save=False,fontsz=14):
    fig, ax = plt.subplots(1,1,dpi=200)
    im1 = ax.imshow(X,cmap='coolwarm')
    for (j,i),label in np.ndenumerate(X):
        ax.text(i,j,round(label,3),ha='center',va='center',fontsize=fontsz)
    ax.set_xticks(np.arange(0,len(X.T)))
    ax.set_yticks(np.arange(0,len(X)))
    ax.set_xticklabels(xtick,fontsize=20,rotation=15)
    ax.set_yticklabels(ytick,fontsize=20)
    if save==True:
        plt.savefig('C:\Kosio\Master_Thesis\Figures\Paper_top_geod\\'+str(save))

def annotate_imshow(D,round_val=2,txt_size=6):
    fig, ax = plt.subplots(1,1,dpi=200)
    ax.imshow(D,aspect='auto')
    for (j,i),label in np.ndenumerate(D):
        if label!=0:
            ax.text(i,j,round(label,round_val),ha='center',va='center',fontsize=txt_size)

def gabor_filter(x,y,theta,lambd):
    x_p = x*np.cos(theta) + y*np.sin(theta)
    y_p = -x*np.sin(theta) + y*np.cos(theta)
    phi = np.exp(-(x_p**2+y_p**2)/2)*np.cos(2*np.pi*(x_p)*lambd)
    return phi

def drifting_grating(x,y,theta,lambd):
    x_p = x*np.cos(theta) + y*np.sin(theta)
    phi = np.cos(2*np.pi*lambd*x_p)
    return phi

def check_mfld(h1,h2,thresh=0.2):
    X1 = np.sum((h1[:,1]-h1[:,0])>thresh)
    X2 = np.sum((h2[:,1]-h2[:,0])>thresh)
    X_bars = np.zeros(5)
    X_bars[0] = np.sum(np.logical_and(X1==0,X2==0))
    X_bars[1] = np.sum(np.logical_and(X1==1,X2==0))
    X_bars[2] = np.sum(np.logical_and(X1==0,X2==1))
    X_bars[3] = np.sum(np.logical_and(X1==2,X2==1))
    X_bars[4] = 1-np.sum(X_bars[:4])
    return X_bars
