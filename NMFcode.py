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

#comparing model to own code
diff = cosine_similarity(W_sk,W_own)
diff1 = cosine_similarity(H_sk,H_own)
print('Cosine similarity between model W and own W:\n',diff)
print('Cosine similarity between model H and own H:\n',diff1)

#comparing model signatures and my signatures to actual signatures
diff2 = cosine_similarity(H_sk,realsig)
print('Cosine similarity between model H and given H:\n',diff2)
diff3 = cosine_similarity(H_own,realsig)
print('Cosine similarity between own H and given H:\n',diff3)

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
    
############ Kasar data ##########
import numpy as np
import scipy as sp
import NMFalgorithm as nmf
from sklearn.decomposition import NMF
import NMFalgorithm as nmf
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)

V = sp.genfromtxt("kasar2015-cll-mutation-matrix.tsv", delimiter="\t")[1:,1:]

#using sklearn
model = NMF(n_components=3,solver='mu')
W = model.fit_transform(V)
H = model.components_

#clustering
kmeans = KMeans(n_clusters=3).fit(H)
signature = kmeans.cluster_centers_
