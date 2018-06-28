import NMFalgorithm as nmf
import numpy as np
import scipy as sp
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

V = sp.genfromtxt("example-mutation-counts.tsv", delimiter="\t")[1:,1:]
realH = np.load("example-signatures.npy")

model = NMF(n_components=5,solver='mu')
W = model.fit_transform(V)
H = model.components_

WH_V_diff = nmf.cosSim(W.dot(H),V)
print(WH_V_diff)
realH_H_diff = cosine_similarity(H,realH)
print(realH_H_diff)
