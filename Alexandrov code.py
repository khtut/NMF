########## Alexandrov data ##########
import numpy as np
import scipy as sp
from sklearn.decomposition import NMF
import NMFalgorithm as nmf
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)

V = sp.genfromtxt("Breast_genomes_mutational_catalog_96_subs.txt", delimiter="\t")[1:,1:]
realsig = sp.genfromtxt("signatures.txt", delimiter="\t")[1:,3:]

#using sklearn
model = NMF(n_components=27,init='random',solver='mu',max_iter=2000)
W = model.fit_transform(V)
H_sk = model.components_

#using own code
W_own,H_own = nmf.nmf(V,27)

#clustering
kmeans = KMeans(n_clusters=27).fit(W)
W_sk = kmeans.cluster_centers_

#comparing model to own code
diff = cosine_similarity(W_sk,W_own)
diff1 = cosine_similarity(H_sk,H_own)
print('Cosine similarity between model W and own W:\n',diff)
print('Cosine similarity between model H and own H:\n',diff1)

#comparing model signatures and my signatures to actual signatures
diff2 = cosine_similarity(W_sk,realsig)
print('Cosine similarity between model W and given W:\n',diff2)
diff3 = cosine_similarity(W_own,realsig)
print('Cosine similarity between own W and given W:\n',diff3)
