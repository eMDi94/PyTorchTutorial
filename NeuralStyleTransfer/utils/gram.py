import torch


def gram_matrix(x):
    # a = batch size (1)
    # b = number of features map
    # (c, d) = dimensions of f. map (N=c*d)
    a, b, c, d = x.size()
    features = x.view(a*b, c*d)

    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)