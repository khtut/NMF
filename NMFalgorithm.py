import numpy as np    

########## Basic NMF ##########

def obj(V,W,H):
    d,n = V.shape
    WH = np.dot(W,H)
    F = (V * np.log(WH) - WH).sum()
    return F

def updateW(V,W,H):
    WH = np.dot(W,H)
    W_new = W * V.dot(np.transpose(H))/WH.dot(np.transpose(H))
    return W_new

def updateH(V,W,H):
    WH = np.dot(W,H)
    H_new = H * (np.transpose(W)).dot(V)/(np.transpose(W)).dot(WH)
    return H_new

def nmf(V,k):
    d,n = V.shape
    W = np.random.rand(d,k)
    H = np.random.rand(k,n)
    F = obj(V,W,H)

    threshold = 1e-5
    maxiteration = 100
    iteration = 0
    converged = False

    while (not converged) and iteration <= maxiteration:
        W_new = updateW(V,W,H)
        H_new = updateH(V,W_new,H)
        F_new = obj(V,W_new,H_new)

        converged = np.abs(F_new-F) <= threshold
        W,H = W_new,H_new
        iteration = iteration + 1

    return W,H

########## nsNMF ##########

def nsobj(V,W,S,H):
    d,n = V.shape
    WSH = W.dot(S).dot(H)
    F = (V * np.log(WSH) - WSH).sum()
    return F

def updateWS(V,W,S,H):
    WH = np.dot(W,H)
    SH = np.dot(S,H)
    W_new = W * np.dot(V / WH, np.transpose(SH))
    W_new = W_new / np.sum(W_new, axis=0, keepdims=True)
    return W_new

def updateHS(V,W,S,H):
    WH = np.dot(W,H)
    WS = np.dot(W,S)
    H_new = H*np.transpose(np.dot(np.transpose(V/WH),WS))
    return H_new

def nsnmf(V,k,theta):
    d,n = V.shape
    W = np.random.rand(d,k)
    H = np.random.rand(k,n)
    S = (1-theta)*np.identity(k) + (theta/k)*np.ones([k,k])
    F = nsobj(V,W,S,H)

    threshold = 1e-5
    maxiteration = 100
    iteration = 0
    converged = False
    
    while (not converged) and iteration <= maxiteration:
        W_new = updateWS(V,W,S,H)
        H_new = updateHS(V,W_new,S,H)
        F_new = nsobj(V,W_new,S,H_new)

        converged = np.abs(F_new-F) <= threshold
        W,H = W_new,H_new
        iteration = iteration + 1

    return W,H
