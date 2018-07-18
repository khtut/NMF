########## Alexandrov data ##########
import numpy as np
import scipy.io
import matlab.engine
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import tsv
np.set_printoptions(suppress=True)

###import Alexandrov data
def importdata(file):
    f = open(file)
    ##find column of mutation types by reading second line (row after column header) and looking for 'A[C>A]A'
    string = f.readlines()[1].rstrip('\n').split('\t')
    col = string.index('A[C>A]A')
    #this is under the assumption that A[C>A]A is the first listed mutation type 
    f.close()
    ##isolate counts after column of mutation types
    f = open(file)
    #not sure how to solve issue of having to close and reopen file to read 2nd line and then read 1st line
    samples = f.readline().rstrip('\n').split('\t')[col+1:] 
    categories = []
    mutation_count = dict()
    for l in f:
        arr = l.rstrip().split()
        cat = arr[col]
        categories.append(cat)
        for count,s in zip(arr[col+1:],samples):
            mutation_count[(cat,s)] = count
    #sample sorting is omitted because signatures are not sorted properly for signatures.txt and is not needed for mutation data
    categories = sorted(categories)

    ##create mutation count matrix
    M = np.array([[mutation_count[(c,s)] for s in samples] for c in categories])
    M = M.astype(np.float)
    return M

a = 'Breast_genomes_mutational_catalog_96_subs.txt'
data = importdata(a)
b = 'signatures.txt'
COSMICsig = importdata(b)
##isolate 5 breast cancer signatures
COSMICsig = COSMICsig[:,[1,2,3,8,13]]

###step 1: dimension reduction
totalmutationsbytype = data.sum(axis=1)
totalmutations = totalmutationsbytype.sum(axis=0)
sortedtotalbytype = np.sort(totalmutationsbytype,axis=0)
condition = np.cumsum(sortedtotalbytype) <= 0.01*totalmutations
rowstoremove = np.argsort(totalmutationsbytype)[np.arange(sum(condition))]
reducedData = np.delete(data,np.argsort(totalmutationsbytype)[np.arange(len(rowstoremove))],axis=0)
scipy.io.savemat('reducedData.mat',{'reducedData': reducedData})

###step 2: bootstrapping
def bootstrap(x):
    res = eng.bootstrap(x)
    res = np.array(res)
    return res

###steps 3, 4: NMF, iterate
numsig = COSMICsig.shape[1]
sig = []
iterations = 500
eng = matlab.engine.start_matlab()
for i in range(iterations):
    M = bootstrap('reducedData.mat')    
    P = np.zeros([M.shape[0],numsig])
    model = NMF(n_components=numsig,init='random',solver='mu',max_iter=5000)
    modelP = model.fit_transform(M)
    ##scale columns of oldP and place results into P
    norm = modelP.sum(axis=0)
    for j in range(modelP.shape[1]):
        P[:,j] = modelP[:,j]/norm[j]
    ##store P into sig
    sig.append(P)
sig = np.array(sig)
##reshape 3d array into 2d array
signatures = np.zeros([M.shape[0],sig.shape[0]*sig.shape[2]])
for i in range(len(sig)):
    signatures[:,np.arange(numsig*i,numsig*(i+1))] = sig[i]

###step 5: cluster
kmeans = KMeans(n_clusters=numsig)
kmeans.fit(np.transpose(signatures))
clustersig = kmeans.cluster_centers_

###insert zeros for removed mutation types
modelsig = np.insert(np.transpose(clustersig),rowstoremove,0,axis=0)

###cosine similarity
diff = cosine_similarity(np.transpose(modelsig),np.transpose(COSMICsig))
diff1 = cosine_similarity(np.transpose(modelsig),np.transpose(modelsig))
diff2 = cosine_similarity(np.transpose(COSMICsig),np.transpose(COSMICsig))
print('Comparing signatures:\n',diff)
print('Comparing elements in produced signatures:\n',diff1)
print('Comparing elements in COSMIC signatures:\n',diff2)
