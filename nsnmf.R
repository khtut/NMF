# install.packages("NMF")
# library(NMF)
# setwd("C:/Users/khpen/OneDrive/Documents/Scholarships, Internships/CBCB")

# import data
V <- read.table(file = 'example-mutation-counts.tsv', sep = '\t', header = TRUE)
V <- V[-1]

# create a NMF object based on random (compatible) matrices
options(scipen=999)
r <- 5
n <- dim(V)[1]
p <- dim(V)[2]
W <- rmatrix(n, r)
H <- rmatrix(r, p)
new('NMFns')
nmfModel(model='NMFns', W=W, H=H)
# random nonsmooth NMF model
model <- nmf(V, r, 'nsNMF')
W <- model@fit@W
H <- model@fit@H
write.table(W, "sampleW.txt", sep="\t", col.names = NA)
write.table(H, "sampleH.txt", sep="\t", col.names = NA)

