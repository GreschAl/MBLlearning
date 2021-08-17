import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True })
plt.rc('font', family='serif')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from MBLlearning.learn_inds.recurrent import get_LSTM_data
from MBLlearning.global_config import get_config
DTYPE, DEVICE = get_config()

##################################################################################################
# dataset
##################################################################################################

class VarianceDataset(Dataset):
    """ Dataset holding sampled disorder vectors with the corresponding h-value and variance as the targets.
        For each system size in L_vals and disorder parameter h in h_grid, samples N_train-times.
        The data is preprocessed by regrouping into subgroups of size seq_len.
    """
    def __init__(self, L_vals, h_grid, N_trains, seq_len):
        self.hvals = []
        self.hcorr = []
        self.targets = []
        
        L_max = max(L_vals)
        for L in L_vals:
            for h in h_grid:
                temp = np.random.uniform(-h,h,size=(N_trains,L))/max(h_grid)
                padded = get_LSTM_data(temp,seq_division=seq_len)
                if L<L_max:
                    L_diff = (L_max - L)
                    # pad zeros s.t. shape(hvals) = (N_train,L,len_seq) --> (N_train,L_max,len_seq)
                    padded = np.pad(padded,((0,0),(0,L_diff),(0,0)),"constant",constant_values=0)
                (self.hvals).append(padded)
                hcorrs = np.full((N_trains,),h)
                (self.hcorr).append(hcorrs)
                (self.targets).append(np.vstack((hcorrs, np.var(temp, axis=-1))))
                
        self.hvals = np.array(self.hvals)
        self.hcorr = np.array(self.hcorr)
        self.targets = np.array(self.targets)
                
        self.L_vals = L_vals
        self.h = h_grid
        self.N = N_trains

    def __len__(self):
        return self.N * len(self.h) * len(self.L_vals)

    def __getitem__(self, idx):
        # transform idx into idx-tuple (L_idx,N)
        L_idx, temp = idx//(len(self.h)*self.N), idx%(len(self.h)*self.N)
        h_idx, N_idx = temp//self.N, temp%self.N
        assert L_idx*len(self.h)*self.N+h_idx*self.N+N_idx == idx, \
        "Index error for idx={}. (L_idx,h_idx,N_idx) were ({},{},{})".format(idx,L_idx,h_idx,N_idx)
        
        hval = self.hvals[L_idx*len(self.h)+h_idx,N_idx]
        target = self.targets[L_idx*len(self.h)+h_idx,:,N_idx]
        return hval, target, self.L_vals[L_idx]
    
##################################################################################################
# rnn-cell
##################################################################################################

class GRUnetSimple(nn.Module):
    """
    GRU cell followed by a fully-connected layer without any activations after the RNN-cell.
    If no hidden_size is provided (by default), no such layer is appended.
    
    Inputs:
    x:  Input sequence. For correct shapes, refer to nn.GRU documentation
    """
    def __init__(self, kernel_size, out_size, hidden_size=None, batch_first=True):
        super().__init__()
        
        if hidden_size is None:
            self.gru = nn.GRU(kernel_size,out_size,batch_first=batch_first)
            self.lin = None
        else:
            self.gru = nn.GRU(kernel_size,hidden_size,batch_first=batch_first)
            self.lin = nn.Linear(hidden_size,out_size)
        
        self.kernel_size = kernel_size
        self.hidden = hidden_size
        self.no_inds = out_size
        
    def forward(self, x):
        _, out = self.gru.forward(x)
        if self.lin is None:
            return out[-1]
        else:
            return self.lin(out[-1])

def train_simple(model,dataloader,epochs=10,learning_rate=1e-3,mute_outputs=False):
    """ Train the model with the provided training data, following a provided optimizer routine.
        mute_outputs suppresses any output created during training.
        Returns the average epoch loss as an array
    """
    loss_curve = []

    model = model.to(device=DEVICE,dtype=DTYPE)
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for e in range(epochs):
        if not mute_outputs:
            print("--- Epoch {} ---".format(e+1))
        epoch_loss = 0.0
        N = 0

        for t, (x, y, x_len) in enumerate(dataloader):                    
            x = x.to(device=DEVICE, dtype=DTYPE)
            y = y.to(device=DEVICE, dtype=DTYPE)

            # make sure that the padded values at the end are not visible for the model
            x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
            scores = model(x)
            scores = scores.view(y.shape)
            loss = F.mse_loss(scores, y, reduction="sum")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            N += len(y)

        loss_curve.append(epoch_loss/N)

        if not mute_outputs:
            print("Avg. epoch loss = {}".format(loss_curve[-1]))

    return model, loss_curve

def eval_model(model,testdata):
    """ Evaluates performance of the model on the given test data loader.
        Returns the average prediction per system size and disorder parameter, as well as the loss
    """

    model.to(device=DEVICE,dtype=DTYPE)
    model.eval()

    means = np.zeros((len(testdata.L_vals),len(testdata.h),2))
    mses  = np.zeros_like(means)

    with torch.no_grad():
        for l,L in enumerate(testdata.L_vals):
            for i,h in enumerate(testdata.h):
                hvals = torch.from_numpy(testdata.hvals[l*len(testdata.h)+i]).to(device=DEVICE,dtype=DTYPE)
                preds = model(hvals).cpu().numpy()
                targs = testdata.targets[l*len(testdata.h)+i].T
                mses[l,i] = ((preds-targs)**2).mean(axis=0)
                means[l,i] = preds.mean(axis=0)
    return means,mses