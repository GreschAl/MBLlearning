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
from MBLlearning.learn_inds.recurrent import LSTMnet, GRUnet, get_lstm_concatenated, get_LSTM_data # defaults
from MBLlearning.learn_inds.linear import default_fc_net
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
    
    def __init__(self,chain_sizes,energies=None):      
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
        
        i_ = 1 if self.energies is None else len(self.energies)
        
        # for each epsilon, provide output neurons for the indicators
        if use_LSTM:
            self.rnn_kernel = LSTMnet(kernel_size,*rnn_params,batch_first=True)
            self.rnn_func = LSTMnet
        else:
            self.rnn_kernel = GRUnet(kernel_size,*rnn_params,batch_first=True)
            self.rnn_func = GRUnet
        self.model = default_fc_net(self.rnn_kernel.hidden_size,no_inds*i_,model_params)
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
        i_ = 1 if self.energies is None else len(self.energies)
        
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
        if model_func is not None:
            assert model_params!=-1, "Provide new parameters (can even be None-type) with the post-processing NN-function."
            self.model = model_func(self.rnn_kernel.hidden_size,self.N_inds*i_,model_params)
            self.model_func = model_func
            self.model_params["parameters"] = model_params
        else:
            if model_params is None:
                params = None
            elif model_params==-1:
                params = self.model_params["parameters"]
            else:
                params = model_params
            self.model = self.model_func(self.rnn_kernel.hidden_size,self.N_inds*i_,params)
            self.model_params["parameters"] = params
        return
            
    
    def _get_train_loader(self,batch_size,seq_division,single_inds=None):
        """ Create the train loader from the train data with a given batch_size """
        return get_lstm_concatenated(self.data,self.Lvals,
                                     batchsize=batch_size,
                                     seq_division=seq_division,
                                     append_chain_size=self.append_chain_size,
                                     single_inds=single_inds
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
            length = self.N_inds if self.energies is None else self.N_inds*len(self.energies)
        else:
            length = data_train["inds"].shape[1]
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

            for t, (x, y, x_len) in enumerate(loader_train):                    
                x = x.to(device=self.DEVICE, dtype=self.DTYPE)
                y = y.to(device=self.DEVICE, dtype=self.DTYPE)

                # make sure that the padded values at the end are not visible for the model
                x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
                scores = model(rnn(x))
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
        
        # since we stack indicators of different energies on top of each other, consider this for param single_inds
        if self.energies is not None:
            i_ = len(self.energies)
            temp = [i in ind_idxs for i in range(self.N_inds)] # produces bit-string whether to keep or not
            ind_idxs = i_*temp # repeat list for all energies
        else:
            i_ = 1
        
        # reset parameters of post-processing model
        self.model = self.model_func(self.rnn_kernel.hidden_size,sum(ind_idxs),self.model_params["parameters"])
        
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
        
        element_wise_loss = torch.zeros(epochs, sum(ind_idxs))
        
        for e in range(epochs):
            if not mute_outputs:
                print("--- Epoch {} ---".format(e+1))
            epoch_loss = 0.0
            N = 0

            for t, (x, y, x_len) in enumerate(loader_train):                    
                x = x.to(device=self.DEVICE, dtype=self.DTYPE)
                y = y.to(device=self.DEVICE, dtype=self.DTYPE)

                # make sure that the padded values at the end are not visible for the model
                x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
                if rnn_opt_params is None:
                    with torch.no_grad():
                        temp = rnn(x)
                    scores = model(temp)
                else:
                    scores = model(rnn(x))
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
        return loss_curve, element_wise_loss
    
    ################ Model evaluation #####################################

    def _predict_data(self,data):
        model = (self.model).to(device=self.DEVICE,dtype=self.DTYPE)
        rnn = (self.rnn_kernel).to(device=self.DEVICE,dtype=self.DTYPE)
        model.eval()
        rnn.eval()
        N_eps = 1 if self.energies is None else len(self.energies)
        
        with torch.no_grad():
            data = data.to(device=self.DEVICE, dtype=self.DTYPE)
            scores = model(rnn(data)).cpu()
        
        scores = np.array(scores.detach())
        return scores

    def predict(self,L,key="test"):
        data = self.data[L][key]
        assert data is not None, "Please provide {} data first via the set_up()-method".format(key)
        seq_division = self.rnn_params["seq_division"]
        h = data["h"]
        hcorr = data["hcorr"]
        hvals = data["h_i"]
        inds = data["inds"]
        predictions = []
        targets = []
        
        for i in h:
            mask = hcorr==i
            data = get_LSTM_data(hvals[mask],seq_division=seq_division,append_chain_size=self.append_chain_size)
            shaping = inds[mask].shape
            targets.append(inds[mask])
            predictions.append(self._predict_data(torch.from_numpy(data)).reshape(*shaping))
            
        return h, np.array(predictions), np.array(targets)
        
    def predict_energy_wise(self,L,key="test"):
        """ Convenience function that calls predict() and sorts the data by energy.
            No averaging or other post-processing is done else.
            If no energies were provided, this function reduces to the predict() function.
        """
        if self.energies is None:
            return self.predict(L,key)
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
            with torch.no_grad():
                estim = model(rnn(hvals)).cpu().numpy()
            self.data[L]["estimation"] = {"h": data["h"], "inds": estim, "h_i": data["h_i"], "hcorr": data["hcorr"], "key": key }
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
    
    def load(self,filename):
        """ Load from file. Typical endings are either .pt or .pth
            Provide a filename ending with '_{}' to load RNN and post-processing NN from seperate files.
        """
        n = filename.find("_{}")
        assert n!=-1, "Please provide a filename containing '_{}'"
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

    
    ################ Cross-validation #####################################

    def cross_validation_mse(self,num_folds=5,epochs=5,batch_size=64):
        """ Does num_fold-cross-validation on the data given an initialized model and its optimizer.
            First, training is done for some epochs, then MSE for each test fold is returned as an array.
        """
        
        folds_mse = []
        seq_division = self.model_params["seq_division"]
        for i in range(num_folds):
            # save the existing train data for reusage
            saved_data = deepcopy(self.data)
            # reserve the i-th data chunck (training data is preshuffled) as validation set
            for L in self.Lvals:
                h, hcorr = self.data[L]["train"]["h"], self.data[L]["train"]["hcorr"]
                x,y = self.data[L]["train"]["h_i"], self.data[L]["train"]["inds"]
                
                X_train_folds = np.array_split(x, num_folds)
                y_train_folds = np.array_split(y, num_folds)
                hc_folds = np.array_split(hcorr, num_folds)
                
                Xte = X_train_folds[i]
                yte = y_train_folds[i]
                hcte = hc_folds[i]
                
                Xtr = np.delete(X_train_folds, obj = i, axis=0)
                Xtr = np.concatenate(Xtr)
                
                ytr = np.delete(y_train_folds, obj = i, axis=0)
                ytr = np.concatenate(ytr)
                
                hctr = np.delete(hc_folds, obj = i, axis=0)
                hctr = np.concatenate(hctr)
                
                self.data[L]["train"] = { "h_i":Xtr, "inds": ytr, "h": h, "hcorr": hctr }
                self.data[L]["test"]  = { "h_i":Xte, "inds": yte, "h": h, "hcorr": hcte }
            
            # train on both chain lengths simultaneously
            self.reset_model()
            self.train(epochs=epochs,mute_outputs=True,batch_size=batch_size)
            mse = 0.0
            for L in self.Lvals:
                Xte, yte = self.data[L]["test"]["h_i"], self.data[L]["test"]["inds"]
                h, hcorr = self.data[L]["test"]["h"], self.data[L]["test"]["hcorr"]
                for j in h:
                    mask = hcorr == j
                    temp = get_LSTM_data(Xte[mask],seq_division=seq_division,append_chain_size=self.append_chain_size)
                    shaping = yte[mask].shape
                    scores = self._predict_data(torch.from_numpy(temp)).reshape(*shaping)
                    mse += ((scores-yte[mask])**2).sum() # can we do this better?
                
            folds_mse.append(mse)
            
            # reset data in model
            self.data = saved_data
        return folds_mse