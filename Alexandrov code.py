########## Alexandrov data ##########
import numpy as np
import scipy.io as sp.io
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import NMFalgorithm as nmfalg
np.set_printoptions(suppress=True)

def import_data(filename):    
    f = open(filename)
    string = f.readlines()[1].rstrip('\n').split('\t')
    col = string.index('A[C>A]A')
    f.close()
    f = open(filename)
    samples = f.readline().rstrip('\n').split('\t')[col+1:] 
    categories = []
    mutation_count = dict()
    for l in f:
        arr = l.rstrip().split()
        cat = arr[col]
        categories.append(cat)
        for count, s in zip(arr[col+1:], samples):
            mutation_count[(cat, s)] = count
    categories = sorted(categories)
    M = np.array([[mutation_count[(c,s)] for s in samples] for c in categories])
    M = M.astype(np.float)
    return M
  
def dimension_reduction(data):
    total_mutations_by_type = data.sum(axis=1)
    total_mutations = total_mutations_by_type.sum(axis=0)
    sorted_total_by_type = np.sort(total_mutations_by_type, axis=0)
    condition = np.cumsum(sorted_total_by_type) <= 0.01*total_mutations
    rows_to_remove = np.argsort(total_mutations_by_type)[np.arange(sum(condition))]
    reduced_data = np.delete(
        data,np.argsort(total_mutations_by_type)[np.arange(len(rows_to_remove))],
        axis=0
        )
    return reduced_data, rows_to_remove
  
def bootstrap(data):
    M = np.array([
        np.random.multinomial(int(round(m_i.sum())), m_i/m_i.sum()) for m_i in data.T
        ])
    return M.T
  
def iterate_nmf(reduced_data, n, iterations):
    sig = []
    for i in range(iterations):
        M = bootstrap(reduced_data)
        P = np.zeros([M.shape[0], n])
        model = NMF(n_components=n, init='random', solver='mu', max_iter=5000)
        model_P = model.fit_transform(M)
        norm = model_P.sum(axis=0)
        for j in range(model_P.shape[1]):
            P[:,j] = model_P[:,j]/norm[j]
        sig.append(P)
    sig = np.array(sig)
    signatures = np.zeros([M.shape[0], sig.shape[0]*sig.shape[2]])
    for i in range(len(sig)):
        signatures[:,np.arange(n*i, n*(i+1))] = sig[i]
    return signatures
  
def kmeans(signatures, n):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(signatures.T) 
    cluster_sig = kmeans.cluster_centers_
    return cluster_sig

def vanilla(a, n, iterations, b=None):
    data = import_data(a)
    reduced_data, rows_to_remove = dimension_reduction(data)
    signatures = iterate_nmf(reduced_data, n, iterations)
    cluster_sig = kmeans(signatures, n)
    model_sig = np.insert(cluster_sig.T, rows_to_remove, 0, axis=0)

    if b != None:
        paper_sig = import_data(b)
        paper_sig = paper_sig[:, [1,2,3,8,13]]     #don't know how to generalize this (extracting certain signatures)
        diff1 = cosine_similarity(model_sig.T, paper_sig.T)
        print('Comparing signatures:\n', diff1)

    return model_sig
