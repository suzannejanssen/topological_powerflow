import numpy as np


def run(graph1=False):
 
    A,W,B = graph()
    # elif graph == graph2:
    #     G = graph2()
    # elif graph == graph3:
    #     G = graph3()
    # else:
    #     G = graph4()
    
    return A,W,B

def graph():
    A = np.matrix([[0, 1, 1, 0, 0], [1, 0, 1, 1, 0], [1, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 0, 1, 0]])
    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1
    w5 = 1

    W = np.matrix([[0, w1, w2, 0, 0], [w1, 0, w3, w4, 0], [w2, w3, 0, 0, 0], [0, w4, 0, 0, w5], [0, 0, 0, w5, 0]])
    # NxL weighted incidence matrix
    B = np.matrix([[w1, w2, 0, 0, 0], [-w1, 0, -w3, w4, 0], [0, -w2, w3, 0, 0], [0, 0, 0, -w4, w5], [0, 0, 0, 0, -w5]])

    return A,W,B
