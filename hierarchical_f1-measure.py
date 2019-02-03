# -*- coding: utf-8 -*-
"""
@author: ugurite
"""


import numpy as np

## Computing F1-score for a hierarchy

def global_Fscore(Z, labels):
    """
    Takes hierarchical clustering Z and true classes 
    Returns Global Fscore as defined in Karypis 2002
    """
    assert(Z.shape[0] == len(labels)-1)
    id_classes = np.unique(labels).tolist()
    Classes = []
    for e in id_classes:
        Classes.append(set(np.where(labels == e)[0].tolist()))       
    # we need to loop on all classes
    S = 0
    N = float(len(labels))
    for set_ in Classes :
        size = len(set_)
        S += HFscore(Z, set_) * (size/N)
    return S


def HFscore(Z, classe):
    """ 
    Takes a linkage matrix and a class
    Returns best Fscore
    """
    n = Z.shape[0]+1
    size = float(len(classe))
    # we need to loop on all the nodes of the hierarchy
    # compute recalls
    Recall_leaves = [int(i in classe) /size for i in range(n)]
    Recall_nodes = []
    for j in range(n-1):
        if Z[j,0] < n :
            r1 = Recall_leaves[Z[j,0]]
        else :
            r1 = Recall_nodes[Z[j,0]-n]
        if Z[j,1] < n :
            r2 = Recall_leaves[Z[j,1]]
        else :
            r2 = Recall_nodes[Z[j,1]-n]
        Recall_nodes.append(r1+r2)
    # compute precisions
    Precision_leaves = [float(i in classe) for i in range(n) ]
    Precision_nodes = []
    for j in range(n-1):
        if Z[j,0] < n :
            p1 = Precision_leaves[Z[j,0]]
            s1 = 1
        else :
            p1 = Precision_nodes[Z[j,0]-n]
            s1 = Z[Z[j,0]-n,3]
        if Z[j,1] < n :
            p2 = Precision_leaves[Z[j,1]]
            s2 = 1
        else :
            p2 = Precision_nodes[Z[j,1]-n]
            s2 = Z[Z[j,1]-n,3]
        Precision_nodes.append( (p1*s1 + p2*s2)/float(s1+s2) )
    ### compute fscores
    Fscores_nodes = []
    for i in range(len(Precision_nodes)):
        if Recall_nodes[i]+Precision_nodes[i] != 0:
            f = (2*Recall_nodes[i]*Precision_nodes[i]) / (Recall_nodes[i]+Precision_nodes[i])
            Fscores_nodes.append(f)
        else :
            Fscores_nodes.append(0)

    return max(Fscores_nodes)