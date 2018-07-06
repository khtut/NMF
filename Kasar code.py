###code to create clustered mutation matrix###
import numpy as np
import pandas as pd

df = pd.read_csv("kasar2015-wgs-cll-mutations.tsv",sep='\t',header=0)

sample = df['Sample']
cat_idx = df['Category Index']
cat = sorted(set(df['Category']))
dist = df['Distance to Nearest Mutation']
samples = sorted(set(df['Sample']))
sample_index = dict()
for i in range(len(df)):
    for index,name in enumerate(samples):
        sample_index[name+' - C'] = index*2
        sample_index[name+' - NC'] = index*2 + 1
    if dist[i] <= 1000:
        sample_idx = sample_index[sample[i]+' - C']
    else:
        sample_idx = sample_index[sample[i]+' - NC']

M = pd.DataFrame(np.zeros([60,96]),index=sample_index,columns=cat)          
clustercounts = np.array([np.zeros(96)])
nonclustercounts = np.array([np.zeros(96)])

for a in range(len(samples)):                                   
    sample_data = df[df['Sample'] == samples[a]]                

    condition = sample_data['Distance to Nearest Mutation'] <= 1000
    for b in list(sample_data.index):
        if condition[b] == True: 
            idx = cat_idx[b]                                    
            clustercounts[0,idx] = clustercounts[0,idx] + 1     
        else: 
            idx = cat_idx[b]                                            
            nonclustercounts[0,idx] = nonclustercounts[0,idx] + 1     
    
    for c in list(sample_data.index):                           
        if condition[c] == True:                                
            M[M.index == samples[a]+' - C'] = list(clustercounts)     
        else:                                                   
            M[M.index == samples[a]+' - NC'] = list(nonclustercounts)      

M.to_csv('kasar2015clustermatrix.txt', header=cat, sep='\t', mode='w')
