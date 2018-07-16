########## simulated data ##########
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import NMFalgorithm as nmf
np.set_printoptions(suppress=True)

V = np.genfromtxt("example-mutation-counts.tsv", delimiter="\t")[1:,1:]
realSig = np.load("example-signatures.npy")

#using sklearn
model = NMF(n_components=5, init='random', solver='mu')
modelW = model.fit_transform(np.transpose(V))
modelH = model.components_

scalemodelW = []
norm1 = modelW.sum(axis=0)
for i in range(modelW.shape[1]):
    scalemodelW.append(modelW[:,i]/norm1[i])
scalemodelW = np.array(scalemodelW)

#using own code
ownW,ownH = nmf.nmf(V,5)

diff = cosine_similarity(scalemodelW, realSig)
print('Comparing produced signature to given signatures:\n', diff)
