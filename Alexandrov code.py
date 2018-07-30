import numpy as np
import scipy.io as sp
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import NMFalgorithm as nmfalg
np.set_printoptions(suppress=True)
import argparse

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
    exp = []
    for i in range(iterations):
        M = bootstrap(reduced_data)
        P = np.zeros([M.shape[0], n])
        model = NMF(n_components=n, init='random', solver='mu', max_iter=1000)
        model_P = model.fit_transform(M)
        model_E = model.components_
        norm = model_P.sum(axis=0)
        for j in range(model_P.shape[1]):
            P[:,j] = model_P[:,j]/norm[j]
        sig.append(P)
        exp.append(model_E)
    sig = np.array(sig)
    exp = np.array(exp)
    model_sig = np.zeros([M.shape[0], sig.shape[0]*sig.shape[2]])
    for i in range(len(sig)):
        model_sig[:,np.arange(n*i, n*(i+1))] = sig[i]
    model_exp = np.zeros([exp.shape[0]*exp.shape[1], M.shape[1]])
    for i in range(len(exp)):
        model_exp[np.arange(n*i, n*(i+1)),:] = exp[i]
    return model_sig, model_exp
  
def kmeans(model_sig, model_exp, n):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(model_sig.T) 
    cluster_sig = kmeans.cluster_centers_
    cluster_exp = []
    for i in range(n):
      new = model_exp[kmeans.labels_==i]
      cluster_exp.append(np.mean(new, axis=0))
    cluster_exp = np.array(cluster_exp)
    return cluster_sig, cluster_exp

def vanilla(a, n, iterations):
    data = import_data(a)
    reduced_data, rows_to_remove = dimension_reduction(data)
    model_sig, model_exp = iterate_nmf(reduced_data, n, iterations)
    cluster_sig, cluster_exp = kmeans(model_sig, model_exp, n)
    signatures = np.insert(cluster_sig.T, rows_to_remove, 0, axis=0)
    exposures = cluster_exp
    return signatures, exposures

####################
#Nonsmooth NMF#
####################

def nonsmooth(reduced_data, n, iterations, theta):
    sig = []
    exp = []
    for i in range(iterations):
        M = bootstrap(reduced_data)
        P = np.zeros([M.shape[0], n])
        model_P, model_E = nmfalg.nsnmf(M, n, theta)     #uses nsNMF code stored in NMFalgorithm.py
        norm = model_P.sum(axis=0)
        for j in range(model_P.shape[1]):
            P[:,j] = model_P[:,j]/norm[j]
        sig.append(P)
        exp.append(model_E)
    sig = np.array(sig)
    exp = np.array(exp)
    model_sig = np.zeros([M.shape[0], sig.shape[0]*sig.shape[2]])
    for i in range(len(sig)):
        model_sig[:,np.arange(n*i, n*(i+1))] = sig[i]
    model_exp = np.zeros([exp.shape[0]*exp.shape[1], M.shape[1]])
    for i in range(len(exp)):
        model_exp[np.arange(n*i, n*(i+1)),:] = exp[i]
    return model_sig, model_exp  
  
def nsnmf(a, n, iterations, theta):
    data = import_data(a)
    reduced_data, rows_to_remove = dimension_reduction(data)
    model_sig, model_exp = nonsmooth(reduced_data, n, iterations, theta)
    cluster_sig, cluster_exp = kmeans(model_sig, model_exp, n)
    signatures = np.insert(cluster_sig.T, rows_to_remove, 0, axis=0)
    exposures = cluster_exp
    return signatures, exposures

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Has functions for vanilla NMF and non smooth NMF ")
    parser.add_argument('-f', '--mutation_file', help="Mutation count data file", required=True)
    parser.add_argument('-n', '--num_sig', help="Number of signatures", required=True)
    parser.add_argument('-m', '--no_iter', help="Number of iterations (default = ", required=False, default = 500)
    parser.add_argument('-flag', '--nmf_flag', help="flag for which NMF to use - van for Vanilla and ns for nonsmooth NMF", required=False, default = "van")
    parser.add_argument('-theta', '--theta_val', help="theta value if using non smooth NMF (default 0.5)", required=False, default = 0.5)
    parser.add_argument('-out', '--out_file', help="Output prefix for storing signature and exposure", required=False, default = "out") 
    args = parser.parse_args()

    if args.nmf_flag =="van":
        signature, exposure = vanilla(args.mutation_file, int(args.num_sig), int(args.no_iter))
    elif args.nmf_flag =="ns":
        signature, exposure = nsnmf(args.mutation_file, int(args.num_sig), int(args.no_iter), args.theta_val)

    
