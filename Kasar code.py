########## Kasar data ##########
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

