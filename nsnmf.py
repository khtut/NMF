import numpy as np
import NMFalgorithm as nmf
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)

V = np.genfromtxt("example-mutation-counts.tsv", delimiter="\t")[1:,1:]
realsig = np.load("example-signatures.npy")

#using R
W_r = np.genfromtxt("sampleW.txt", delimiter="\t")[1:,1:]
H_r = np.genfromtxt("sampleH.txt", delimiter="\t")[1:,1:]

#using own code
W_own,H_own = nmf.nsnmf(V,5,0.5)

diff = cosine_similarity(np.transpose(W_r),np.transpose(W_own))
print('Comparing W matrices of R results and own results:\n',diff)
diff1 = cosine_similarity(H_r,H_own)
print('Comparing H matrices of R results and own results:\n',diff1)

diff2 = cosine_similarity(H_r,realsig)
print('Comparing H matrix from R to given signatures:\n',diff2)
diff3 = cosine_similarity(H_own,realsig)
print('Comparing H matrix from own code to given signatures:\n',diff3)
