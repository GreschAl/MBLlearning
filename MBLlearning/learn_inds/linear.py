import torch.nn as nn

############## Fully-connected post-processing network #############################
    
def default_fc_net(feature_size, num_inds, hidden_sizes=None):
    """ Default fully-connected network with ReLU-activations of arbitrary hidden layer number.
        Must provide input size (feature_size) and final output size (num_inds).
        If any hidden_sizes are provided (not by default), adds the corresponding hidden layers in between.
    """
    if hidden_sizes is None:
        net = nn.Linear(feature_size,num_inds)
    else:
        temp = []
        h_prev = feature_size
        for h in hidden_sizes:
            temp.append(nn.Linear(h_prev,h))
            temp.append(nn.ReLU())
            h_prev = h
        temp.append(nn.Linear(h_prev,num_inds))
        net = nn.Sequential(*temp)
    return net