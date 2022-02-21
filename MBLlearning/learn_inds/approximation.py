##################################################################################################
# Defines a class holding the model (a three-layer FC-net by default) and the necessary data for
# training and evaluating.
# The model can be changed after set-up, but has to allow a mapping from an input vector
# from disorder targets _h_ = (h1,h2,..,hL)  ---> _i_ = (i1,i2,...,iN) to indicator values
# The chain lengths L or [L1,L2,...] and the number of indicators N is fixed for this class upon initialization
##################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import sampler

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# for handling datasets with variable sequence lengths

import torch.nn.functional as F

import numpy as np
from MBLlearning.utils.data import load_training_data # for data loading and preprocessing
from MBLlearning.learn_inds.recurrent import LSTMnet, GRUnet, get_lstm_concatenated, get_LSTM_data, get_lstm_energies
from MBLlearning.learn_inds.linear import default_fc_net
from MBLlearning.learn_inds.conv import get_conv_loader
from MBLlearning.global_config import get_config

from copy import deepcopy

############## Approximation class for variable chain lengths ######################################

class Approximator():
    """ Class holding both the models (rnn_kernel and post-processing model) as well as the necessary data
        for training and evaluation.
        The range of the training data is specified by providing the corresponding chain_sizes as an array.
        Can also learn multiple energy values at once by concatenating the corresponding targets to single, larger
        target for regression. Concatenation occurs energy-wise. To do so, provide an array for energies, defaults to False
    """
    
    def __init__(self,chain_sizes,energies=None,add_energy_density=False,eps_shift=0,eps_scale=1):      
        self.rnn_kernel = None
        self.rnn_func = None
        self.rnn_params = {}
        self.model = None
        self.model_func = None
        self.model_params = {}
        
        self.optimizer = None
        self.opt_params = None
        
        self.Lvals = chain_sizes
        self.energies = energies
        self.sorting = False # for internal usage
        self.N_inds = None
        
        self.dangling_net = False # for internal usage
        self.no_danglings = 0
        
        # if set to true, promotes the energy density (shifted by eps_shift) to a feature after the rnn kernel
        self.add_energy_density = False if energies is None else add_energy_density
        self.eps_shift = eps_shift
        self.eps_scale = eps_scale
        
        self.data = {}
        for L in chain_sizes:
            self.data[L] = { "train": None, "test": None, "estimation": None, "parameters": None }
        
        DTYPE, DEVICE = get_config()
        self.DTYPE = DTYPE
        self.DEVICE = DEVICE

    ############## Model initialization ######################################

    def set_up(self,fileloc,rnn_params=(10,),model_params=None,seq_division=None,
                      no_params=2,use_LSTM=False,learn_single_ind=None,N_train=None,append_chain_size=False):
        """ Sets up model (GRU model by default) and test/training data for all given chain lengths.
            Data is filled in as dictionaries as data[L][key]. Fileloc has to be of the form
            '*_L_{}*.txt' with optional further placeholders after this first one
            If append_chain_size is activated (not by default), the chain size is appended to the input stream
            of the recurrent network during training and prediction.
        """
        n = fileloc.find("_L_")
        assert n!=-1, "Please provide a filename containing '_L_'"
        # this guarantuees applicability in both cases of energy provided or not
        
        self.append_chain_size = append_chain_size
        
        for L in self.Lvals:
            filename = fileloc[:n+6].format(L)+fileloc[n+6:]
            hcorr_train, hvals_train, inds_train,hcorr_test,hvals_test, inds_test, no_inds, params = load_training_data(
                filename,
                no_params=no_params,
                N_train=N_train,
                energies=self.energies,
                single_inds=learn_single_ind,
                sorting=self.sorting)
            
            if L==self.Lvals[-1]:
                print("Using",no_inds,"indicators:")
                for indic in params["indicators"][2:]:
                    print(indic)
            self.N_inds = no_inds
            
            self.data[L]["train"] = {"h_i":hvals_train, "inds": inds_train,
                                  "hcorr": hcorr_train, "h": np.sort(np.unique(hcorr_train))}
            self.data[L]["test"]  = {"h_i":hvals_test, "inds": inds_test,
                                  "hcorr": hcorr_test, "h": np.sort(np.unique(hcorr_test))}
            self.data[L]["parameters"] = params

        temp = {"parameters": rnn_params}
        
        kernel_size = 1 if seq_division is None else seq_division
        
        if self.append_chain_size:
            # append an input neuron for the chain size
            kernel_size += 1
        
        i_ = 1 if self.add_energy_density else len(self.energies)
        
        # for each epsilon, provide output neurons for the indicators
        if use_LSTM:
            self.rnn_kernel = LSTMnet(kernel_size,*rnn_params,batch_first=True)
            self.rnn_func = LSTMnet
        else:
            self.rnn_kernel = GRUnet(kernel_size,*rnn_params,batch_first=True)
            self.rnn_func = GRUnet
        feature_size = self.rnn_kernel.hidden_size + 1 if self.add_energy_density else self.rnn_kernel.hidden_size
        self.model = default_fc_net(feature_size,no_inds*i_,model_params)
        self.model_func = default_fc_net
        self.model_params["parameters"] = model_params
        self.model_params["fileloc"] = fileloc
        temp["seq_division"] = seq_division
        temp["fileloc"] = fileloc
        self.rnn_params = temp
        return
    
    def reset_model(self,model_func=None,model_params=-1,rnn_func=None,rnn_params=None):
        """ If provided, initialize the RNN with the given model and its parameters (kernel_size, hidden sizes etc only)
            Otherwise, reset the already used RNN with a new initialization and with new parameters if provided.
            Does likewise for the post-processing model
        """
        i_ = 1 if self.add_energy_density else len(self.energies)
        
        # reset the RNN
        if rnn_func is not None:
            assert rnn_params is not None, "Provide new parameters with the RNN-function."
            if self.append_chain_size:
                rnn_params = list(rnn_params)
                rnn_params[0] += 1
            self.rnn_kernel = rnn_func(*rnn_params,batch_first=True)
            self.rnn_func = rnn_func
            self.rnn_params["parameters"] = rnn_params[1:]
            self.rnn_params["seq_division"] = rnn_params[0]
            if self.append_chain_size:
                self.rnn_params["seq_division"] -= 1
        else:
            params = self.rnn_params["parameters"] if rnn_params is None else rnn_params[1:]
            kernel_size = self.rnn_params["seq_division"] if rnn_params is None else rnn_params[0]
            if self.append_chain_size:
                # append an input neuron for the chain size
                kernel_size += 1
            self.rnn_kernel = self.rnn_func(kernel_size,*params,batch_first=True)
            self.rnn_params["parameters"] = params
            self.rnn_params["seq_division"] = kernel_size
            if self.append_chain_size:
                self.rnn_params["seq_division"] -= 1
                
        # reset the post-processing model
        feature_size = self.rnn_kernel.hidden_size + 1 if self.add_energy_density else self.rnn_kernel.hidden_size
        if model_func is not None:
            assert model_params!=-1, "Provide new parameters (can even be None-type) with the post-processing NN-function."
            self.model = model_func(feature_size,self.N_inds*i_,model_params)
            self.model_func = model_func
            self.model_params["parameters"] = model_params
        else:
            if model_params is None:
                params = None
            elif model_params==-1:
                params = self.model_params["parameters"]
            else:
                params = model_params
            self.model = self.model_func(feature_size,self.N_inds*i_,params)
            self.model_params["parameters"] = params
        return
            
    
    def _get_train_loader(self,batch_size,seq_division,single_inds=None):
        """ Create the train loader from the train data with a given batch_size """
        sort_energy = None if not self.add_energy_density else self.sort
        if seq_division == "conv":
            return get_conv_loader(self.data,self.Lvals,batchsize=batch_size,single_inds=single_inds)
        if seq_division == "features":
            return get_lstm_energies(self.data,self.Lvals,
                                     self.energies,
                                     batchsize      = batch_size,
                                     single_inds    = single_inds,
                                     sort_by_energy = sort_energy,
                                     eps_shift      = self.eps_shift,
                                     eps_scale      = self.eps_scale
                                    )
        return get_lstm_concatenated(self.data,self.Lvals,
                                     batchsize=batch_size,
                                     seq_division=seq_division,
                                     append_chain_size=self.append_chain_size,
                                     single_inds=single_inds,
                                     sort_by_energy=sort_energy,
                                     energies=self.energies
                                    )
    
    def swap_datasets(self):
        """ Exchanges training data with the test data """
        for L in self.Lvals:
            temp_test  = self.data[L]["train"].copy()
            temp_train = self.data[L]["test"].copy()
            self.data[L]["train"] = temp_train
            self.data[L]["test"]  = temp_test
        return

    ################## Model training procedure ########################################################

    def train(self,epochs=10, batch_size=64,mute_outputs=False,single_inds=None,L_tests=None):
        """ Train the model with the provided training data, following a provided optimizer routine.
            mute_outputs suppresses any output created during training.
            If single_inds is provided (not by default), use only those indicators for training.
            Returns the average epoch loss as an array
            If L_tests are provided as an array (not by default), also tracks the epoch loss for the test set
        """
        
        data_train = self.data[self.Lvals[0]]["train"]
        assert data_train is not None, "Please provide train data first via the set_up()-method"
        opt_func = self.optimizer
        assert opt_func is not None, "Please provide an optimizer first"
        
        N_inds = self.N_inds
        if single_inds is not None:
            # change output layer of post-processing model
            self.N_inds = len(single_inds)
            self.reset_model()
            length = self.N_inds if self.energies is None or self.add_energy_density else self.N_inds*len(self.energies)
        else:
            length = data_train["inds"].shape[1]//len(self.energies) if self.add_energy_density else data_train["inds"].shape[1]
        model = (self.model).to(device=self.DEVICE,dtype=self.DTYPE)
        rnn = (self.rnn_kernel).to(device=self.DEVICE,dtype=self.DTYPE)
        model.train()
        rnn.train()
        # since we stack indicators of different energies on top of each other, consider this for param single_inds
        if self.energies is not None and single_inds is not None:
            temp = [i in single_inds for i in range(N_inds)] # produces bit-string whether to keep or not
            single_inds = len(self.energies)*temp # repeat list for all energies
        
        loader_train = self._get_train_loader(batch_size,self.rnn_params["seq_division"],single_inds=single_inds)

            
        optimizer = opt_func(model.parameters(),*self.opt_params)
        optimizer_rnn = opt_func(rnn.parameters(),*self.opt_params)
        
        loss_curve = []
        element_wise_loss = torch.zeros(epochs, length)
        losses = []
        
        for e in range(epochs):
            if not mute_outputs:
                print("--- Epoch {} ---".format(e+1))
            epoch_loss = 0.0
            N = 0

            for t, data in enumerate(loader_train):
                if self.add_energy_density:
                    x, y, x_len, eps = data
                    eps = eps - self.eps_shift
                    eps = (self.eps_scale*eps).to(device=self.DEVICE, dtype=self.DTYPE)
                else:
                    x, y, x_len = data
                x = x.to(device=self.DEVICE, dtype=self.DTYPE)
                y = y.to(device=self.DEVICE, dtype=self.DTYPE)

                if self.rnn_params["seq_division"] == "conv":
                    scores = model(rnn.train_forward(x,x_len))
                else:
                    # make sure that the padded values at the end are not visible for the model
                    x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
                    scores = rnn(x)
                    if self.add_energy_density:
                        scores = torch.cat((scores,eps),dim=1)
                    scores = model(scores)
                    scores = scores.view(y.shape)

                
                loss = F.mse_loss(scores, y, reduction="sum")

                with torch.no_grad():
                    # compute indicator- and energy-wise loss
                    temp = ((scores-y)**2).sum(dim=0).detach().cpu()
                    element_wise_loss[e,:] += temp

                optimizer.zero_grad()
                optimizer_rnn.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_rnn.step()

                epoch_loss += loss.item()
                N += len(y)

            loss_curve.append(epoch_loss/N)
            element_wise_loss[e] /= N
            
            if L_tests is not None:
                with torch.no_grad():
                    temp_losses = [ [],[] ] # one for train, one for test in l-dependence
                    for i,key in enumerate(["train","test"]):
                        for l in L_tests:
                            ins = self.data[l][key]["h_i"]
                            ins = torch.from_numpy( get_LSTM_data(ins,seq_division=self.rnn_params["seq_division"],
                                                                  append_chain_size=self.append_chain_size) )
                            
                            preds = model(rnn(ins.to(device=self.DEVICE,dtype=self.DTYPE)))
                            outs = torch.from_numpy( self.data[l][key]["inds"] ).to(device=self.DEVICE,dtype=self.DTYPE)

                            loss = F.mse_loss(preds, outs, reduction="mean")
                            temp_losses[i].append( loss.item() )
                losses.append(temp_losses)

            if not mute_outputs:
                print("Avg. epoch loss = {}".format(loss_curve[-1]))
                    
        
        self.model = model
        self.rnn_kernel = rnn
        self.N_inds = N_inds # reset counter
        if L_tests is None:
            return loss_curve, element_wise_loss
        else:
            return loss_curve, np.array(losses).transpose([1,0,2]) # data key x N_epochs x L
    
    def train_transfer(self,ind_idxs,rnn_opt_params=None, epochs=10, batch_size=64,mute_outputs=False):
        """ Takes the previously trained model + RNN and trains on the newly chosen indicator(s) ind_idxs
            following a provided optimizer routine.
            Before training starts, the model's parameters are reset automatically.
            By default, the RNN is frozen during training. rnn_opt_params can be provided for fine-tuning, however.
            mute_outputs suppresses any output created during training.
            Returns the average epoch loss as an array
        """
        loss_curve = []
        
        data_train = self.data[self.Lvals[0]]["train"]
        assert data_train is not None, "Please provide train data first via the set_up()-method"
        opt_func = self.optimizer
        assert opt_func is not None, "Please provide an optimizer first"
        
        N_inds = self.N_inds
        if ind_idxs is not None:
            # change output layer of post-processing model
            self.N_inds = len(ind_idxs)
        
        # since we stack indicators of different energies on top of each other, consider this for param single_inds
        if self.energies is not None and ind_idxs is not None:
            i_ = len(self.energies)
            temp = [i in ind_idxs for i in range(N_inds)] # produces bit-string whether to keep or not
            ind_idxs = i_*temp # repeat list for all energies
        else:
            i_ = 1
        
        # reset parameters of post-processing model
        num_features = self.rnn_kernel.hidden_size
        if self.add_energy_density:
            num_features += 1
            # temp does not need to be overwritten
        else:
            temp = ind_idxs
        self.model = self.model_func(num_features,sum(temp),self.model_params["parameters"])
        
        model = (self.model).to(device=self.DEVICE,dtype=self.DTYPE)
        rnn = (self.rnn_kernel).to(device=self.DEVICE,dtype=self.DTYPE)
        model.train()
        if rnn_opt_params is None:
            rnn.eval()
        else:
            rnn.train()
            optimizer_rnn = opt_func(rnn.parameters(),*rnn_opt_params)
        
            
        loader_train = self._get_train_loader(batch_size,self.rnn_params["seq_division"],single_inds=ind_idxs)
        optimizer = opt_func(model.parameters(),*self.opt_params)
        
        element_wise_loss = torch.zeros(epochs, sum(temp))
        
        for e in range(epochs):
            if not mute_outputs:
                print("--- Epoch {} ---".format(e+1))
            epoch_loss = 0.0
            N = 0

            for t, data in enumerate(loader_train):
                if self.add_energy_density:
                    x, y, x_len, eps = data
                    eps = eps - self.eps_shift
                    eps = (self.eps_scale*eps).to(device=self.DEVICE, dtype=self.DTYPE)
                else:
                    x, y, x_len = data
                
                x = x.to(device=self.DEVICE, dtype=self.DTYPE)
                y = y.to(device=self.DEVICE, dtype=self.DTYPE)

                # make sure that the padded values at the end are not visible for the model
                x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
                if rnn_opt_params is None:
                    with torch.no_grad():
                        scores = rnn(x)
                else:
                    scores = rnn(x)
                if self.add_energy_density:
                    scores = torch.cat((scores,eps),dim=1)
                scores = model(scores)
                scores = scores.view(y.shape)
                
                loss = F.mse_loss(scores, y, reduction="sum")

                with torch.no_grad():
                    # compute indicator- and energy-wise loss
                    temp = ((scores-y)**2).sum(dim=0).detach().cpu()
                    element_wise_loss[e,:] += temp

                if rnn_opt_params is None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    optimizer_rnn.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer_rnn.step()

                epoch_loss += loss.item()
                N += len(y)

            loss_curve.append(epoch_loss/N)
            element_wise_loss[e] /= N

            if not mute_outputs:
                print("Avg. epoch loss = {}".format(loss_curve[-1]))
                    
        
        self.model = model
        if rnn_opt_params is None:
            self.rnn_kernel = rnn
        self.N_inds = N_inds # reset counter
        return loss_curve, element_wise_loss
    
    def train_from_features(self, epochs=10, batch_size=64, single_inds=None, mute_outputs=False,L_tests=None,every_N=1):
        """ Takes the previously RNN and trains the consecutive model only
            following a provided optimizer routine.
            Before training starts, the model's parameters are reset automatically.
            The RNN is frozen during training.
            mute_outputs suppresses any output created during training.
            If L_tests is provided, tracks the epoch- and L-dependent loss during training.
            Returns the average epoch loss as an array
        """
        loss_curve = []

        data_train = self.data[self.Lvals[0]]["train"]
        assert data_train is not None, "Please provide train data first via the set_up()-method"
        assert data_train.get("features") is not None, "Please provide features first via the get_features-method"
        if L_tests is not None:
            losses = []
            has_attribute = self.data[L_tests[0]]["test"].get("features") is not None
            assert has_attribute,"Please provide features for the test data first via the get_features-method"
            # prepare data once for all epochs
            feats_train, feats_test = [], []
            for key,listing in zip(["train","test"],(feats_train,feats_test)):
                for l in L_tests:
                    inds  = self.data[l][key]["inds"]
                    feats = self.data[l][key]["features"]
                    if inds.shape[1] == len(self.energies):
                        inds = inds.T[:,:,np.newaxis]
                    else:
                        inds = self.sort(inds)
                    # gather the corresponding energy densities in one list
                    in_energies = ( np.ones((len(self.energies),inds.shape[1])) * self.energies[:,np.newaxis] ).reshape((-1,1))
                    inds = inds.reshape((-1,inds.shape[-1]))
                    # repeat the input data for each energy in energies
                    feats = np.repeat(feats.reshape((1,*feats.shape)),len(self.energies),axis=0)
                    feats = feats.reshape((-1,*feats.shape[2:]))
                    # rescale energy and put together with features
                    in_energies = self.eps_scale*(in_energies - self.eps_shift)
                    ins = np.hstack((feats,in_energies))
                    listing.append([ins,inds])
        opt_func = self.optimizer
        assert opt_func is not None, "Please provide an optimizer first"

        N_inds = self.N_inds
        if single_inds is not None:
            # change output layer of post-processing model
            self.N_inds = len(single_inds)
            self.reset_model()
            length = self.N_inds if self.energies is None or self.add_energy_density else self.N_inds*len(self.energies)
        else:
            length = data_train["inds"].shape[1]//len(self.energies) if self.add_energy_density else data_train["inds"].shape[1]
        model = (self.model).to(device=self.DEVICE,dtype=self.DTYPE)
        rnn = (self.rnn_kernel).to(device=self.DEVICE,dtype=self.DTYPE)
        model.train()
        rnn.train()
        # since we stack indicators of different energies on top of each other, consider this for param single_inds
        if self.energies is not None and single_inds is not None:
            temp = [i in single_inds for i in range(N_inds)] # produces bit-string whether to keep or not
            single_inds = len(self.energies)*temp # repeat list for all energies

        model = (self.model).to(device=self.DEVICE,dtype=self.DTYPE)
        model.train()

        loader_train = self._get_train_loader(batch_size,"features",single_inds=single_inds)
        optimizer = opt_func(model.parameters(),*self.opt_params)

        element_wise_loss = torch.zeros(epochs, length)

        for e in range(epochs):
            if not mute_outputs:
                print("--- Epoch {} ---".format(e+1))
            epoch_loss = 0.0
            N = 0

            for t, (x,y) in enumerate(loader_train):
                # since we already work with the output of the RNN, x already represents the scores
                scores = x.to(device=self.DEVICE, dtype=self.DTYPE)
                y = y.to(device=self.DEVICE, dtype=self.DTYPE)

                scores = model(scores)
                scores = scores.view(y.shape)

                loss = F.mse_loss(scores, y, reduction="sum")

                with torch.no_grad():
                    # compute indicator- and energy-wise loss
                    temp = ((scores-y)**2).sum(dim=0).detach().cpu()
                    element_wise_loss[e,:] += temp

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                N += len(y)

            loss_curve.append(epoch_loss/N)
            element_wise_loss[e] /= N
            
            if L_tests is not None:
                if e%every_N==0:
                    with torch.no_grad():
                        temp_losses = [ [],[] ] # one for train, one for test in l-dependence
                        for i,(key,listing) in enumerate(zip(["train","test"],(feats_train,feats_test))):
                            for l,lis in zip(L_tests,listing):
                                ins, outs = lis
                                preds = model(torch.from_numpy(ins).to(device=self.DEVICE,dtype=self.DTYPE))
                                outs = torch.from_numpy( outs ).to(device=self.DEVICE,dtype=self.DTYPE)

                                loss = F.mse_loss(preds, outs, reduction="mean")
                                temp_losses[i].append( loss.item() )
                    losses.append(temp_losses)

            if not mute_outputs:
                print("Avg. epoch loss = {}".format(loss_curve[-1]))

        self.model = model
        self.N_inds = N_inds # reset counter
        if L_tests is None:
            return loss_curve, element_wise_loss
        else:
            return loss_curve, np.array(losses).transpose([1,0,2]) # data key x N_epochs x L
    
    ################ Model evaluation #####################################
    
    def get_features(self,L,key="test"):
        """ Returns the output of the RNN kernel for all <key> data. """
        rnn = self.rnn_kernel.to(device=self.DEVICE,dtype=self.DTYPE)
        rnn.eval()
        dic = self.data[L][key]
        h, hcorr, hvals = dic["h"], dic["hcorr"], dic["h_i"]
        inputs = get_LSTM_data(hvals,seq_division=self.rnn_params["seq_division"])
        inputs = torch.tensor(inputs).to(device=self.DEVICE,dtype=self.DTYPE)
        
        with torch.no_grad():
            outs = rnn.forward(inputs).cpu().numpy()
            
        return h,hcorr,outs

    def _predict_data(self,data,eps=None):
        model = (self.model).to(device=self.DEVICE,dtype=self.DTYPE)
        rnn = (self.rnn_kernel).to(device=self.DEVICE,dtype=self.DTYPE)
        model.eval()
        rnn.eval()
        
        with torch.no_grad():
            data = data.to(device=self.DEVICE, dtype=self.DTYPE)
            if eps is not None:
                eps = eps - self.eps_shift
                eps = (self.eps_scale*eps).to(device=self.DEVICE, dtype=self.DTYPE)
                scores = model(torch.cat((rnn(data),eps),dim=1)).cpu()
            else:
                scores = model(rnn(data)).cpu()
        
        scores = np.array(scores.detach())
        return scores

    def predict(self,L,key="test",eps=None):
        data = self.data[L][key]
        assert data is not None, "Please provide {} data first via the set_up()-method".format(key)
        seq_division = self.rnn_params["seq_division"]
        h = data["h"]
        hcorr = data["hcorr"]
        hvals = data["h_i"]
        inds = data["inds"]
        
        predictions = []
        targets = []
        
        if eps is not None:
            inds = self.sort(inds)
            for e,inds_temp in zip(self.energies,inds):
                temp_pred, temp_targ = [], []
                for i in h:
                    mask = hcorr==i
                    data = get_LSTM_data(hvals[mask],seq_division=seq_division,append_chain_size=self.append_chain_size)
                    shaping = inds_temp[mask].shape
                    temp_targ.append(inds_temp[mask])
                    temp_pred.append(self._predict_data(torch.from_numpy(data),
                                                        eps=torch.full((np.sum(mask),1),e) # shift etc is applied later
                                                        ).reshape(*shaping))
                predictions.append(np.array(temp_pred))
                targets.append(np.array(temp_targ))
        else:
            for i in h:
                mask = hcorr==i
                data = get_LSTM_data(hvals[mask],seq_division=seq_division,append_chain_size=self.append_chain_size)
                shaping = inds[mask].shape
                targets.append(inds[mask])
                predictions.append(self._predict_data(torch.from_numpy(data)).reshape(*shaping))
            
        return h, np.array(predictions), np.array(targets)
        
    def predict_energy_wise(self,L,key="test"):
        """ Convenience function that calls predict() and sorts the data by energy.
            Returns the tuple (h, prediction, target) sorted by energy, repeats, indicator and h-value respectively.
            No averaging or other post-processing is done else.
            If no energies were provided, this function reduces to the predict() function.
        """
        if self.energies is None:
            return self.predict(L,key)
        elif self.add_energy_density:
            return self.predict(L,key,self.energies)
        else:
            h, scores, targets = self.predict(L,key)
            scores  = self.sort(scores)
            targets = self.sort(targets)
            return h, scores, targets

    
    def estimate(self,key="test"):
        """ Run the key data through the network and save the output as "estimation""<key-val>" for later usage """
        seq_division = self.rnn_params["seq_division"]
        model = (self.model).to(device=self.DEVICE,dtype=self.DTYPE)
        rnn = (self.rnn_kernel).to(device=self.DEVICE,dtype=self.DTYPE)
        model.eval()
        rnn.eval()
        
        for L in self.Lvals:
            data = self.data[L][key]
            assert data is not None, "Please provide {} data first via the set_up()-method".format(key)
            hvals = torch.from_numpy(get_LSTM_data(data["h_i"],seq_division=seq_division,
                                                   append_chain_size=self.append_chain_size)
                                    ).to(device=self.DEVICE,dtype=self.DTYPE)
            
            if self.add_energy_density:
                estim = []
                length = len(hvals)
                for eps in self.energies:
                    temp = self._predict_data(hvals,eps=torch.full((length,1),eps)) # shift etc is applied later
                    estim.append(temp)
                estim = np.array(estim) # has shape N_eps x N_data x N_inds
                # transform into N_data x (N_inds * N_eps)
                estim = estim.transpose([1,0,2]).reshape(length,-1)
            else:
                with torch.no_grad():
                    estim = model(rnn(hvals)).cpu().numpy()
            self.data[L]["estimation"] = {"h": data["h"], "inds": estim, "h_i": data["h_i"], "hcorr": data["hcorr"], "key": key }
        return
    
    def clear_estimates(self):
        for L in self.Lvals:
            self.data[L]["estimation"] = None
            
    def fix_features(self,key="test"):
        for l in self.Lvals:
            _, _, feats = self.get_features(l,key)
            self.data[l][key]["features"] = feats
        return
    
    ############## utility functions ######################################
    
    def save(self,filename):
        """ Save to file. Typical endings are either .pt or .pth.
            Provide a filename ending with '_{}' to save RNN and post-processing NN seperately.
        """
        n = filename.find("_{}")
        assert n!=-1, "Please provide a filename containing '_{}'"
        torch.save(self.model.state_dict(), filename.format("model"))
        torch.save(self.rnn_kernel.state_dict(), filename.format("rnn"))
        return
    
    def load(self,filename,only_rnn=False):
        """ Load from file. Typical endings are either .pt or .pth
            Provide a filename ending with '_{}' to load RNN and post-processing NN from seperate files.
            If only_rnn is selected (not by default), only loads the RNN from file.
        """
        n = filename.find("_{}")
        assert n!=-1, "Please provide a filename containing '_{}'"
        if not only_rnn:
            self.model.load_state_dict(torch.load(filename.format("model"),map_location=self.DEVICE))
        self.model.eval()
        self.rnn_kernel.load_state_dict(torch.load(filename.format("rnn"),map_location=self.DEVICE))
        self.rnn_kernel.eval()
        return
    
    def sort(self,data):
        """ If energies were provided, re-sorts the data energy-wise """
        if self.energies is None:
            return data
        if self.sorting:
            data = np.array( [ data[...,i+np.arange(self.N_inds)*len(self.energies)] for i in range(len(self.energies))] )
        else:
            data = np.array( [ data[...,self.N_inds*i:self.N_inds*(i+1)] for i in range(len(self.energies))] )
        return data
    
    def num_params(self):
        """ Returns the number of parameters for the RNN kernel and the consecutive model """
        # RNN
        num_params_rnn = 0
        for key,val in self.rnn_kernel.state_dict().items():
            temp = len(val.cpu().view(-1).numpy())
            num_params_rnn += temp
        print("Total number of parameters in RNN-kernel:\t{}".format(num_params_rnn))
        # model
        num_params_model = 0
        for key,val in self.model.state_dict().items():
            temp = len(val.cpu().view(-1).numpy())
            num_params_model += temp
        print("Total number of parameters in model:\t\t{}".format(num_params_model))
        print("Total number of all parameters:\t\t\t{}".format(num_params_model+num_params_rnn))
        return
