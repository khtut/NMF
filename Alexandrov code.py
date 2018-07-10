########## Alexandrov data ##########
from scipy.io import loadmat
import numpy as np
import collections
from sklearn.decomposition import NMF
#import NMFalgorithm as nmf
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)

oldM = np.genfromtxt('Breast_genomes_mutational_catalog_96_subs.txt',delimiter='\t')[1:,1:]
realsig = np.genfromtxt("signatures.txt", delimiter="\t")[1:,3:]
realsig = realsig[:,[1,2,3,8,13]]
numsig = realsig.shape[1]

#step 1: dimension reduction
totalmutationsbytype = oldM.sum(axis=1)
totalmutations = totalmutationsbytype.sum(axis=0)
sortedtotalbytype = np.sort(totalmutationsbytype,axis=0)
condition = np.cumsum(sortedtotalbytype) <= 0.01*totalmutations
numberrowstoremove = len(np.extract(condition,sortedtotalbytype))
M = np.delete(oldM,np.argsort(totalmutationsbytype)[np.arange(numberrowstoremove)],axis=0)

#step 3 and 4: NMF, iterate
sig = []
iterations = 5
for i in range(iterations):
    P = np.zeros([M.shape[0],numsig])
    model = NMF(n_components=numsig,init='random',solver='mu',max_iter=1000)
    oldP = model.fit_transform(M)
    norm = oldP.sum(axis=0)
    for j in range(oldP.shape[1]):
        P[:,j] = oldP[:,j]/norm[j]
    sig.append(P)
sig = np.array(sig)
signatures = np.zeros([M.shape[0],sig.shape[0]*sig.shape[2]])
for i in range(len(sig)):
    signatures[:,np.arange(len(sig)*i,len(sig)*(i+1))] = sig[i]

#step 5: cluster
kmeans = KMeans(n_clusters=numsig)
kmeans.fit(np.transpose(signatures))
clustersig = kmeans.cluster_centers_

#cosine similarity
diff = cosine_similarity(clustersig,np.transpose(realsig))
