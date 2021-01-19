import numpy as np
import scipy

import torch
from torch import nn as nn



class LogisticRegression(nn.Module):
    def __init__(self, args, num_classes):
        super(LogisticRegression, self).__init__()
        self.dense = nn.Linear(args.adv_units, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_state):
        out = self.dense(hidden_state)
        out = self.sigmoid(out)
        return out
    
    def get_weights(self):
        # return coef as numpy array
        dense_parameter = {name:param for name, param in self.dense.named_parameters()}
        # get coef and covert to numpy
        return dense_parameter["weight"].detach().cpu().numpy()

def get_rowspace_projection(W):
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T) # orthogonal basis
    
    w_basis = w_basis * np.sign(w_basis[0][0]) # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace
    
    return P_W

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices, input_dim):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf: 
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param dim: input dim
    """
    
    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis = 0)
    P = I - get_rowspace_projection(Q)
    
    return P

def hidden_representation_projection(Proj_Matrix, input_hidden):
    return (Proj_Matrix.dot(input_hidden.T)).T 