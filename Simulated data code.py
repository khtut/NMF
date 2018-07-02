########## simulated data ##########
import numpy as np
import scipy as sp
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import NMFalgorithm as nmf
np.set_printoptions(suppress=True)

V = sp.genfromtxt("example-mutation-counts.tsv", delimiter="\t")[1:,1:]
realsig = np.load("example-signatures.npy")

#using sklearn
model = NMF(n_components=5,solver='mu')
W_sk = model.fit_transform(V)
H_sk = model.components_

#using own code
W_own,H_own = nmf.nmf(V,5)
##
###comparing model to own code
##diff = cosine_similarity(W_sk,W_own)
##diff1 = cosine_similarity(H_sk,H_own)
##print('Cosine similarity between model W and own W:\n',diff)
##print('Cosine similarity between model H and own H:\n',diff1)
###comparing model signatures and my signatures to actual signatures
##diff2 = cosine_similarity(H_sk,realsig)
##print('Cosine similarity between model H and given H:\n',diff2)
##diff3 = cosine_similarity(H_own,realsig)
##print('Cosine similarity between own H and given H:\n',diff3)
##  
