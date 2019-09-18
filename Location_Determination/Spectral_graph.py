# Building matrices for power graph
# Topology used is from Topology.JPG

import numpy as np

#A = np.matrix([[0, 1, 1, 0, 0], [1, 0, 1, 1, 0], [1, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 0, 1, 0]])
#Hale's A matrix
A = np.matrix([[0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 1, 0]])

# admittance of each lines (1/reactance), adjust if necessary
# w1 = 1/2
# w2 = 1/4
# w3 = 1/10
# w4 = 1/20
# w5 = 1/0.5

L=7 #number of links
w1 = 1
w2 = 1
w3 = 1
w4 = 1
w5 = 1
w6 = 1
w7 = 1

#W = np.matrix([[0, w1, w2, 0, 0], [w1, 0, w3, w4, 0], [w2, w3, 0, 0, 0], [0, w4, 0, 0, w5], [0, 0, 0, w5, 0]])
#Hale's W matrix
W = np.matrix([[0, w1, 0, 0, 0, 0], [w1, 0, w3, w4, w5, 0], [0, w2, 0, w5, 0, 0], [0, w3, w5, 0, 0, w6], [0, w4, 0, 0, 0, w7], [0, 0, 0, w6, w7, 0]])
# NxL weighted incidence matrix
#B = np.matrix([[w1, w2, 0, 0, 0], [-w1, 0, -w3, w4, 0], [0, -w2, w3, 0, 0], [0, 0, 0, -w4, w5], [0, 0, 0, 0, -w5]])
#Hale's B matrix
B = np.matrix([[w1, 0, 0, 0, 0, 0, 0], [-w1, w2, w3, w4, 0, 0, 0], [0, -w2, 0, 0, w5, 0, 0], [0, 0, -w3, 0, -w5, w6, 0], [0, 0, 0, -w4, 0, 0, w7], [0, 0, 0, 0, 0, -w6, -w7]])
# Power input per node
P = np.matrix([[2], [1], [0], [-1.5], [-1.5], [0]])
N = W.shape[0]

def diagonal(W):
    """Make a diagonal matrix D from W. 
    Each diagonal entry is formed by sum of components of W of the corresponding row. 
    Returns diagonal matrix D"""

    # Axis is 1 to sum over the rows
    rowsum = W.sum(axis=1)

    # squeeze to 1-D numpy array
    rowsum = np.asarray(rowsum).squeeze()

    # lay out over diagonal
    D = np.diag(rowsum)

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

def laplacian(w,v):
    """Returns the inverse of a matrix, based on its reduced eigenvectors and eigenvalues."""

    w_recip = np.reciprocal(w)
    N = len(w_recip)
    w_inv = np.zeros((N,N))
    for i in range(N):
        w_inv[i][i] = w_recip[i] 
    
    Qinv = np.linalg.multi_dot([v, w_inv, v.transpose()])

    return Qinv

def new_laplacian(i, j, Qinv, omega, w):
    """New laplacian calculated for the link addition between node i and j. 
    Returns the new laplacian. Q'inv
    --------------------------
    i : node from
    j : node to
    Qinv : old laplacian
    omega : effective resistance matrix
    w : weight of the added link"""

    N = Q.shape[0]

    e = np.subtract(e(N,i), e(N,j))
    x = np.linalg.multi_dot([Qinv, e, e.transpose(), Qinv])
    scale = w/(1+(omega.item(j,i)*w))
    Qinv_new = np.subtract(Qinv, np.multiply(scale,x))

    return Qinv_new

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

# #Pseudoinverse function for checking
# Qpinv = np.linalg.pinv(Q, rcond=1e-10)

# Pseudo inverse of the laplacian, when reduced=True
w,v = svd(Q,reduced=True,test=False)
Qinv = laplacian(w,v)

#Flow in the graph per link
F = np.linalg.multi_dot([B.transpose(),Qinv,P])
#print(A)
# print(np.where(A==0)[0], np.where(A==0)[1])


def get_nodes_lines(A, exist=True):
    """From adjacency matrix A, it finds the nodes between which there exists or does not exists a line.
    Returns the coordinates in a list of tuples. 
    -----------
    A : adjacency matrix of a graph
    """
    if not exist:
        A = -(A - 1)

    # get upper triangles with ones
    uptr_ones = np.tril(A, -1).T

    # get boolean coordinates matrix
    coord_matrix = np.multiply(A, uptr_ones)

    # extract coordinates and rezip to list of coordinates
    coordinates = np.where(coord_matrix==1)
    coordinates = list(zip(*coordinates))

    return coordinates


def omega(Qinv): 
    """Get Omega from the pseudoinverse. 
    Omega = z*u^T + u*z^T - 2*Qinv
    --------------
    Qinv : the pseudoinverse"""
    
    N = Qinv.shape[0]
    z = np.zeros((N,1))
    for i in range(N):
        z[i] = Qinv.item((i,i))
    
    u = np.ones((N,1))

    omega1 = np.add(np.multiply(z, u.transpose()), np.multiply(u, z.transpose()))
    omega = np.add(omega1, np.multiply(-2, Qinv))

    return omega
    

def delta_flow(omega, A, W, L):
    """Builts the deltaf matrix from figure 2.2 (thesis hale) considering f_ij is unity. """
    new_row_coord = [0, 0, 0, 1, 2, 2, 2]
    new_col_coord = [1, 4, 5, 5, 3, 4, 5]

    new_line_coordinates = list(zip(new_row_coord, new_col_coord))
    # new_line_coordinates = get_nodes_lines(A, exist=False)
    exist_line_coordinates = get_nodes_lines(A, exist=True)

    omega = omega(Qinv)
    deltaf = np.zeros((len(new_line_coordinates), len(exist_line_coordinates)))

    for x, (i, j) in enumerate(new_line_coordinates):
        for y, (a, b) in enumerate(exist_line_coordinates):
            delta = (W[b,a] * (omega[i,a]-omega[j,a]+omega[j,b]-omega[i,b]))/2
            deltaf[x,y] = delta

    return deltaf

delta_f = delta_flow(omega, A, W, L)

np.set_printoptions(suppress=True, precision=4)
print(delta_f)