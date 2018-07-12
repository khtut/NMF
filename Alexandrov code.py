########## Alexandrov data ##########
import numpy as np
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)

def importdata(file):
    f = open(file)

    #find column of mutation types by reading second line (row after column header) and looking for 'A[C>A]A'
    string = f.readlines()[1].rstrip('\n').split('\t')
    col = string.index('A[C>A]A')
    f.close()

    #isolate counts after column of mutation types
    f = open(file)
    #alternate way to read 2nd line and then read 1st line with having to close and reopen file?
    samples = f.readline().rstrip('\n').split('\t')[col+1:] 
    categories = []
    mutation_count = dict()
    for l in f:
        arr = l.rstrip().split()
        cat = arr[col]
        categories.append(cat)
        for count,s in zip(arr[col+1:],samples):
            mutation_count[(cat,s)] = count
    ##samples = sorted(samples)
    #sample sorting is omitted because signatures are not sorted properly for signatures.txt and is not needed for mutation data
    categories = sorted(categories)
    
    #create mutation count matrix
    M = np.array([[mutation_count[(c,s)] for s in samples] for c in categories])
    M = M.astype(np.float)
    return M

#import Alexandrov data
a = 'Breast_genomes_mutational_catalog_96_subs.txt'
oldM = importdata(a)
b = 'signatures.txt'
realsig = importdata(b)
realsig = realsig[:,[1,2,3,8,13]]       #isolate 5 breast cancer signatures

#step 1: dimension reduction
totalmutationsbytype = oldM.sum(axis=1)
totalmutations = totalmutationsbytype.sum(axis=0)
sortedtotalbytype = np.sort(totalmutationsbytype,axis=0)
condition = np.cumsum(sortedtotalbytype) <= 0.01*totalmutations
rowstoremove = np.argsort(totalmutationsbytype)[np.arange(sum(condition))]
M = np.delete(oldM,np.argsort(totalmutationsbytype)[np.arange(len(rowstoremove))],axis=0)

#steps 3, 4: NMF, iterate
numsig = realsig.shape[1]
sig = []
iterations = 10
for i in range(iterations):
    P = np.zeros([M.shape[0],numsig])
    model = NMF(n_components=numsig,init='random',solver='mu',max_iter=1000)
    oldP = model.fit_transform(M)
    #scale columns of oldP and place results into P
    norm = oldP.sum(axis=0)
    for j in range(oldP.shape[1]):
        P[:,j] = oldP[:,j]/norm[j]
    #store P into sig
    sig.append(P)
sig = np.array(sig)
#reshape 3d array into 2d array
signatures = np.zeros([M.shape[0],sig.shape[0]*sig.shape[2]])
for i in range(len(sig)):
    signatures[:,np.arange(numsig*i,numsig*(i+1))] = sig[i]

#step 5: cluster
kmeans = KMeans(n_clusters=numsig)
kmeans.fit(np.transpose(signatures))
clustersig = kmeans.cluster_centers_

#insert zeros for removed mutation types
modelsig = np.insert(np.transpose(clustersig),rowstoremove,0,axis=0)

#cosine similarity
diff = cosine_similarity(np.transpose(modelsig),np.transpose(realsig))
diff1 = cosine_similarity(np.transpose(modelsig),np.transpose(modelsig))
print('Comparing signatures:\n',diff)
print('Comparing elements in produced signatures:\n',diff1)
