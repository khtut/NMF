import numpy as np
import NMFalgorithm as nmf
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)

V = np.genfromtxt("example-mutation-counts.tsv", delimiter="\t")[1:,1:]
realsig = np.load("example-signatures.npy")

########## basic NMF comparison ##########

#using sklearn
model = NMF(n_components=5,init='random',max_iter=5000)
modelW = model.fit_transform(V)
modelH = model.components_

#using own code
W_own,H_own = nmf.nmf(V,5)

diff = cosine_similarity(np.transpose(modelW),np.transpose(W_own))
print('Comparing W matrices of sklearn results and own results:\n',diff)
diff1 = cosine_similarity(modelH,H_own)
print('Comparing H matrices of sklearn results and own results:\n',diff1)

diff2 = cosine_similarity(modelH,realsig)
print('Comparing H matrix from sklearn to given signatures:\n',diff2)
diff3 = cosine_similarity(H_own,realsig)
print('Comparing H matrix from own code to given signatures:\n',diff3)

########## nsNMF comparison ##########

#using R
W_r = np.genfromtxt("sampleW.txt", delimiter="\t")[1:,1:]
H_r = np.genfromtxt("sampleH.txt", delimiter="\t")[1:,1:]

#using own code
W_own,H_own = nmf.nsnmf(V,5,0.5)

diff = cosine_similarity(np.transpose(W_r),np.transpose(W_own))
print('Comparing nonsmooth W matrices of R results and own results:\n',diff)
diff1 = cosine_similarity(H_r,H_own)
print('Comparing nonsmooth H matrices of R results and own results:\n',diff1)

diff2 = cosine_similarity(H_r,realsig)
print('Comparing nonsmooth H matrix from R to given signatures:\n',diff2)
diff3 = cosine_similarity(H_own,realsig)
print('Comparing nonsmooth H matrix from own code to given signatures:\n',diff3)
