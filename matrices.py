# Building matrices for power graph
# Topology used is from Topology.JPG

import numpy as np

A = np.matrix([[0, 1, 1, 0, 0], [1, 0, 1, 1, 0], [1, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 0, 1, 0]])

# admittance of each lines (1/reactance), adjust if necessary
# w1 = 1/2
# w2 = 1/4
# w3 = 1/10
# w4 = 1/20
# w5 = 1/0.5

w1 = 1
w2 = 1
w3 = 1
w4 = 1
w5 = 1

W = np.matrix([[0, w1, w2, 0, 0], [w1, 0, w3, w4, 0], [w2, w3, 0, 0, 0], [0, w4, 0, 0, w5], [0, 0, 0, w5, 0]])

N = W.shape[0]

def diagonal(W):
    """Make a diagonal matrix D from W. 
    Each diagonal entry is formed by sum of components of W of the corresponding row. 
    Returns diagonal matrix D"""

    rowsum = W.sum(axis=1)    #Axis is 1 to sum over the rows
    D = np.zeros((W.shape[0], W.shape[0]))

    for i in range(W.shape[0]):
        D[i][i] = rowsum[i]

    return D

D = diagonal(W)

Q = np.subtract(D,W)



# See description of numpy.linalg.eigh for details about what is returned exactly
w,v = np.linalg.eigh(Q)
w = w[1:N]
v = v[:,1:N]
one = np.matmul(v[:,0].transpose(), v[:,0])
print(one)
# print(v[:,0])