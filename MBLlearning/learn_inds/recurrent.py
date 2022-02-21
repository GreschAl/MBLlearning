import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
# for creating and merging datasets

from torch.utils.data import sampler

import torch.nn.functional as F

import numpy as np
from MBLlearning.utils.data import load_training_data # for data loading and preprocessing
from MBLlearning.global_config import get_config



############## LSTM / GRU kernels #############################

class LSTMnet(nn.Module):
    """
    Easy GRU set-up. Takes as input three integers and a boolean,
    kernel_size: length of the input for each cell
    hidden_size: feature size of LSTM-cell
    num_blocks: specify how many cells to use. Defaults to 1
    batch_first (bool): specifies the required shape of the input. See nn.GRU documentation for details. Defaults to False

    Forward pass optional argument:
    h0c0: Initial value for (hidden_state h0, cell_state c0). If not provided, default to torch default (all zeros)
    """
    def __init__(self, kernel_size, hidden_size, num_blocks=1, batch_first=False):
        super().__init__()
        self.rnn  = nn.LSTM(kernel_size,hidden_size,num_blocks,batch_first=batch_first)
        
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        
    def forward(self, x, h0c0 = None):
        output, (hn, cn) = self.rnn(x) if h0c0 is None else self.rnn(x,h0c0)
        self.output = output
        self.hn = hn
        self.cn = cn
        # take only the last output from the last block as output
        return hn[-1]
    
class GRUnet(nn.Module):
    """
    Easy GRU set-up. Takes as input three integers and a boolean,
    kernel_size: length of the input for each cell
    hidden_size: feature size of LSTM-cell
    num_blocks: specify how many cells to use. Defaults to 1
    batch_first (bool): specifies the required shape of the input. See nn.GRU documentation for details. Defaults to False

    Forward pass optional argument:
    h0: Initial value for (hidden_state h0, cell_state c0). If not provided, default to torch default (all zeros)
    """ 
    def __init__(self, kernel_size, hidden_size, num_blocks=1, batch_first=False):
        super().__init__()
        self.rnn = nn.GRU(kernel_size,hidden_size,num_blocks,batch_first=batch_first)
        
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        
    def forward(self, x, h0=None):
        hn, out = self.rnn(x) if h0 is None else self.rnn(x,h0)
        self.output = out
        self.hn = hn
        return out[-1]
    
class GRUnetSerial(nn.Module):
    """
    GRU set-up
    The network can be duplicated in length where the second RNN cell receives the output of the first one as
    its initialization. Each cell can be transformed into its stacked version as well.
    Takes as input four integers and two booleans:
    kernel_size: length of the input for each cell
    num_reps: specify how many cells are glued to each other in length. Defaults to 1 which correponds to 'GRUnet' above
    num_blocks: specify how stacked the cells are in depth. Defaults to 1
    batch_first (bool): specifies the required shape of the input. See nn.GRU documentation for details. Defaults to False
    allow_skips (bool): inserts skip connections after each rnn cell past the first one. Defaults to False
    
    Inputs:
    x:  Input sequence. For correct shapes, refer to nn.GRU documentation
    h0: The initial hidden state of the GRU cell. If not provided, use torch defaults (all zeros)
    """
    def __init__(self, kernel_size, hidden_size, num_reps=1,
                 num_blocks=1, allow_skips=False, batch_first=False):
        super().__init__()
        self.rnn = nn.ModuleList([nn.GRU(kernel_size,hidden_size,num_blocks,
                                         batch_first=batch_first) for i in range(num_reps)])
        
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_reps = num_reps
        self.batch_first = int(not batch_first) # switch true and false for shape assignment later
        self.allow_skips = allow_skips
        
    def forward(self, x, h0=None):
        _, out = (self.rnn[0]).forward(x) if h0 is None else (self.rnn[0]).forward(x,h0)
        if self.allow_skips:
            # only add changes onto the previous hidden state
            for i in range(1,self.num_reps):
                h0 += out[-1].view(*h0.shape)
                _, out = (self.rnn[i]).forward(x,h0)
        else:
            # overwrite hidden state with each repetition
            for i in range(1,self.num_reps):
                h0 = out[-1].view(1,*(out[-1].shape))
                _, out = (self.rnn[i]).forward(x,h0)
        return out[-1]

############### Further data preparation for RNN cells ################
def _rollout(array,n):
    N = len(array)
    assert n <= N, "Choice of n={} not useful. Was bigger than array size of {}".format(n,N)
    out = []
    array = np.append(array,array)
    for i in range(N):
        out.append(array[i:i+n])
    return out

def append_chain_size_to_data(data):
    """ Helper function for get_LSTM_data.
        Appends the corresponding L value in each partition whose value can be inferred from the data shape.
    """
    shape = [i for i in data.shape]
    shape[-1] = 1 # only append L once on the last axis
    return np.append(data,np.ones(shape)*shape[1],axis=-1)

def get_LSTM_data(data,seq_division=None,append_chain_size=False):
    """ Transform input vector [h1, h2, h3, ... ] into [ [h1,h2,h3], [h2,h3,h4], ... ] + periodic boundary conditions.
        Size of the division is specified by seq_division. If not provided, default to 1, i.e. [ [h1], [h2], ... ]
        If append_chain_size is True (not by default), appends the corresponding chain length to data as well
        If seq_division is set to "conv", a simple reshape is performed
    """
    if seq_division == "conv":
        return data[...,np.newaxis,:]
    N = data.shape[1]
    if seq_division is None or seq_division==1:
        out = data.reshape((-1,N,1))
    elif seq_division == 2:
        out = []
        for temp in data:
            out.append(np.split(np.roll(np.repeat(temp,2),-1),(np.arange(2,2*N,2))))
            # repeat each element and bring first element to last position, then divide into stacks of 2
        out = np.array(out)
    else:
        out = []
        for temp in data:
            out.append(_rollout(temp,seq_division))
        out = np.array(out)
    if append_chain_size:
        out = append_chain_size_to_data(out)
    return out

def load_lstm_TensorDataset(data,L,Lmax=-1,seq_division=None,key="train",
                            dangling_params=None,append_chain_size=False,
                            single_inds=None,sort_by_energy=None,energies=None):
    """ Provide data from a dictionary to be transformed into data readible by a recurrent net.
        If single_inds is provided (not by default), puts only those indicators into the data loader.
        if Lmax is provided, check whether the given L is smaller and append 0 at the sequence end
        based on the trick by
        https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch
        -is-good-for-your-health-61d35642972e
    """
    hvals_train = data[key]["h_i"]
    inds_train  = data[key]["inds"]
    
    if single_inds is not None:
        # data is of form N_samples x N_inds
        inds_train = inds_train[:,single_inds]
        if len(single_inds)==1:
            inds_train = inds_train.reshape((-1,1)) # for training compability, keep second axis
    lstm_train = get_LSTM_data(hvals_train,seq_division=seq_division,append_chain_size=append_chain_size)
    
    if sort_by_energy is not None:
        # append energy-wise if a sorting function is provided and len(single_inds)!=1
        if inds_train.shape[1] == len(energies):
            inds_train = inds_train.T[:,:,np.newaxis]
        else:
            inds_train = sort_by_energy(inds_train)
        # gather the corresponding energy densities in one list
        in_energies = ( np.ones((len(energies),inds_train.shape[1])) * energies[:,np.newaxis] ).reshape((-1,1))
        inds_train = inds_train.reshape((-1,inds_train.shape[-1]))
        # repeat the input data for each energy in energies
        lstm_train = np.repeat(lstm_train.reshape((1,*lstm_train.shape)),len(energies),axis=0)
        lstm_train = lstm_train.reshape((-1,*lstm_train.shape[2:]))
    
    
    print(key,"data shape before padding:",lstm_train.shape)
    # trick comes here
    if L<Lmax:
        L_diff = (Lmax - L)
        # pad zeros s.t. shape(hvals) = (N_train,L,len_seq) --> (N_train,L_max,len_seq)
        lstm_train = np.pad(lstm_train,((0,0),(0,L_diff),(0,0)),"constant",constant_values=0)
        print("L changed from",L,"to",Lmax)
        #L = Lmax # only temporarily redefined for the in_len definition
    print(key,"data shape after padding:",lstm_train.shape)
    
    inputs = torch.tensor(lstm_train)
    target = torch.tensor(inds_train)
    in_len = torch.tensor(np.ones((len(lstm_train)))*L,dtype=torch.int32)
    if dangling_params is None:
        if sort_by_energy is not None:
            data_train = TensorDataset(inputs,target,in_len,torch.tensor(in_energies))
        else:
            data_train = TensorDataset(inputs,target,in_len)
    else:
        hcorr_train = data[key]["hcorr"]
        hmin, hmax = dangling_params["hmin"], dangling_params["hmax"]

        # if hcorr <= hmin, assign label 0 for deloc.
        # if hcorr >= hmax, assign label 1 for MBL
        # else              assign label 2 for None
        label_train = np.ones_like(hcorr_train)*2
        label_train[hcorr_train<=hmin] = 0
        label_train[hcorr_train>=hmax] = 1
        
        label  = torch.tensor(label_train,dtype=torch.int32)

        data_train = TensorDataset(inputs,target,in_len,label)
    
    return data_train, len(inds_train)

def get_lstm_concatenated(data,L_vals,batchsize=64,seq_division=None,key="train",
                          dangling_params=None,append_chain_size=False,single_inds=None,
                          sort_by_energy=None,energies=None):
    """ Constructs one loader of the data from all the specified chain lengths.
        If single_inds is provided (not by default), puts only those indicators into the data loader.
    """
    datalist_train = []
    #N_train = 0
    L_max   = max(L_vals)
    for L in L_vals:
        assert data[L][key] is not None, "Please provide {} data first!".format(key)
        temp_train, len_train = load_lstm_TensorDataset(data[L],L,L_max,seq_division,key,
                                                        dangling_params,append_chain_size,
                                                        single_inds,sort_by_energy=sort_by_energy,energies=energies)
        datalist_train.append(temp_train)
        #N_train += len_train
    data_train = ConcatDataset(datalist_train)
    
    loader_train = DataLoader(data_train, batch_size=batchsize, shuffle=True)
    return loader_train

def get_lstm_energies(data,L_vals,energies,batchsize=64,key="train",single_inds=None,
                          sort_by_energy=None,eps_shift=0,eps_scale=1):
    """ Constructs one loader of the data from all the specified chain lengths.
        If single_inds is provided (not by default), puts only those indicators into the data loader.
    """
    datalist_train = []
    for L in L_vals:
        assert data[L][key] is not None, "Please provide {} data first!".format(key)
        temp_train, _ = load_lstm_energies_TensorDataset(data[L],energies,eps_shift,eps_scale,key,
                                                         single_inds,sort_by_energy=sort_by_energy)
        datalist_train.append(temp_train)
    data_train = ConcatDataset(datalist_train)
    
    loader_train = DataLoader(data_train, batch_size=batchsize, shuffle=True)
    return loader_train

def load_lstm_energies_TensorDataset(data,energies,shift,scale,key="train",single_inds=None,sort_by_energy=None):
    """ Provide data from a dictionary to be transformed into data readible by the neural net.
        If single_inds is provided (not by default), puts only those indicators into the data loader.
    """
    lstm_train = data[key]["features"]
    inds_train  = data[key]["inds"]
    
    if single_inds is not None:
        # data is of form N_samples x N_inds
        inds_train = inds_train[:,single_inds]
        if len(single_inds)==1:
            inds_train = inds_train.reshape((-1,1)) # for training compability, keep second axis
    
    if sort_by_energy is not None:
        # append energy-wise if a sorting function is provided and len(single_inds)!=1
        if inds_train.shape[1] == len(energies):
            inds_train = inds_train.T[:,:,np.newaxis]
        else:
            inds_train = sort_by_energy(inds_train)
        # gather the corresponding energy densities in one list
        in_energies = ( np.ones((len(energies),inds_train.shape[1])) * energies[:,np.newaxis] ).reshape((-1,1))
        inds_train = inds_train.reshape((-1,inds_train.shape[-1]))
        # repeat the input data for each energy in energies
        lstm_train = np.repeat(lstm_train.reshape((1,*lstm_train.shape)),len(energies),axis=0)
        lstm_train = lstm_train.reshape((-1,*lstm_train.shape[2:]))
        
        in_energies = scale*(in_energies - shift)
        lstm_train = np.hstack((lstm_train,in_energies))
    
    inputs = torch.tensor(lstm_train)
    target = torch.tensor(inds_train)
    data_train = TensorDataset(inputs,target)
    
    return data_train, len(inds_train)

def get_data_from_loader(loader):
    ins, outs, lens = next(iter(loader))
    skip = True
    for i,o,l in loader:
        if skip:
            skip=False
            continue
        ins  = torch.cat((ins,i),0)
        outs = torch.cat((outs,o),0)
        lens = torch.cat((lens,l),0)
    return (ins,outs,lens)