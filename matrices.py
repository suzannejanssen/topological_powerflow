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
# NxL weighted incidence matrix
B = np.matrix([[w1, w2, 0, 0, 0], [-w1, 0, -w3, w4, 0], [0, -w2, w3, 0, 0], [0, 0, 0, -w4, w5], [0, 0, 0, 0, -w5]])
# Power input per node
P = np.matrix([[2], [1], [0], [-1.5], [-1.5]])
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

def svd(Q, reduced=False, test=False):
    """Gives the the spectral decomposition of matrix Q. If test=True, it will output the matrix
    Q - v^T*diag(w)*v. If all the values are (close to) zero, the spectral decomposition has succeeded.
    If reduced is true, it returns the v and w with the smallest eigenvalue and its corresponding 
    eigenvector left out.  
    Output: 
    w : the eigenvalues in ascending order
    v : matrix of eigenvectors. column v[:, i] is the normalized eigenvector corresponding to eigenvalue w[i]"""

    # See description of numpy.linalg.eigh for details about what is returned exactly
    w,v = np.linalg.eigh(Q)
    N = len(w)

    if test == True:

        # Create matrix with eigenvalues on the diagonal. 
        eigv = np.zeros((N, N))
        for i in range(N):
            eigv[i][i] = w[i]
        
        Q2 = np.linalg.multi_dot([v, eigv, v.transpose()])
        Q4 = np.subtract(Q,Q2)
        print('Matrix Q minus v * w * v^T should be (approaching) the zero matrix, which is:', Q4 )

    if reduced == True:
        eigv = w[1:N]
        w = eigv
        v = v[:,1:N]  # +1 because the new w is reduced

    return w,v

def inverse(w,v):
    """Returns the inverse of a matrix, based on its reduced eigenvectors and eigenvalues."""

    w_recip = np.reciprocal(w)
    N = len(w_recip)
    w_inv = np.zeros((N,N))
    for i in range(N):
        w_inv[i][i] = w_recip[i] 
    
    Qinv = np.linalg.multi_dot([v, w_inv, v.transpose()])

    return Qinv

def e(size, index):
    """Returns the base vector. 
    ------------
    size : length of the base vector
    index : index at which base vector contains a one, note: index starts from 0 in python """
    arr = np.zeros(size)
    arr[index] = 1.0

    return arr

D = diagonal(W)

Q = np.subtract(D,W)

#Pseudoinverse function
Qpinv = np.linalg.pinv(Q, rcond=1e-10)

# Pseudo inverse of the laplacian, when reduced=True
w,v = svd(Q,reduced=True,test=False)
Qinv = inverse(w,v)

#Flow in the graph per link
F = np.linalg.multi_dot([B.transpose(),Qinv,P])
#print(A)
#print(np.where(A==0)[0], np.where(A==0)[1])

def get_nodes_newlines(A):
    """From adjacency matrix A, it finds the nodes between which there exists no line.
    Returns the row and column coordinates separately. 
    -----------
    A : adjacency matrix of a graph"""

    N = A.shape[0]

    #Create all ones matrix
    ones = np.ones((N,N))
    #Make it an upper triangular matrix excluding diagonal
    uptr_ones = np.triu(ones, k=1)
    #Make it a lower triangular matrix including diagonal
    lowtr_ones = np.tril(ones, k=0)
    #Multiply A by upper triangular matrix, to only get usefull information. 
    #Add with lower triangular matrix, in order to only find necessary zero entries. 
    coord_matrix = np.add(np.multiply(A,uptr_ones), lowtr_ones) 
    row_coord = np.where(coord_matrix==0)[0]
    col_coord = np.where(coord_matrix==0)[1]

    return row_coord,col_coord

