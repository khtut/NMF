########## simulated data ##########
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import NMFalgorithm as nmf
np.set_printoptions(suppress=True)

M = np.genfromtxt("example-mutation-counts.tsv", delimiter="\t")[1:,1:]
realSig = np.load("example-signatures.npy")

#using sklearn
model = NMF(n_components=5, init='random', solver='mu',max_iter=500)
modelW = model.fit_transform(M)
modelH = model.components_

#using own code
ownW,ownH = nmf.nmf(M,5)

#comparing signatures
diff = cosine_similarity(modelH, realSig)
print('Comparing sklearn produced signature to given signatures:\n', diff)

diff2 = cosine_similarity(ownH, realSig)
print('Comparing own implementation produced signature to given signatures:\n', diff2)
