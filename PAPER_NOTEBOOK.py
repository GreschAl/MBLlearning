#!/usr/bin/env python
# coding: utf-8

# This notebook showcases the usage of the `MBLlearning`-package.
# It loads all necessary data and the pretrained model(s) but can also incorporate training the model from scratch.
# All follow-up code is used to recreate every plot from the result section in the main text of the arXiv-paper


# Load relevant packages, in particular, the MBLlearning-package
# load pytorch for training
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# standard libraries for data processing and plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from MBLlearning.learn_inds.approximation import Approximator
from MBLlearning.learn_inds import recurrent
import MBLlearning.utils as utils
import MBLlearning.utils.plotting as plot
from MBLlearning.global_config import get_config, diction
DTYPE, DEVICE = get_config()
NUM_DEVICES = torch.cuda.device_count() if torch.cuda.is_available() else 1
print("Using {} device(s) of type {}".format(NUM_DEVICES,DEVICE))

################################################################################################################
################################################################################################################
# all relevant hyperparameters and data setting go here
################################################################################################################
################################################################################################################
# data settings
fileloc = "data/train_test_data/indicators_train_Uniform_L_{}_eps_{}.txt"
L = [10,12,14]
L_small = [10,12] # for quantitative extrapolation testing
L_extra = [16,18] # for qualitative  extrapolation testing
eps = np.array([0.5]) # pick one out of np.arange(1,20)/20
eps_show = eps # select subset of indicators for which to show plots
single_inds = [0,3,5]
######## transfer learning indicators #######
init_inds = [0,1]
transfer_inds = [2]
#############################################
sorting = False
N_train = 1000
#############################################
# switches for various subsections and what to show
show_single_model  = False
show_features      = False
show_coefficient   = False
show_transfer      = True
show_phase_diag    = False
show_N_train_curve = False
################################################################################################################
# hyperparameters of the model
feature_size = 2 # single number for RNN feature extractor
depth = 1 # depth of RNN-cell, see documentation of nn.GRU or nn.LSTM
use_LSTM = False # switch between GRU or LSTM cell
use_energy_feature = False #True
energy_shift = 0.5
hidden_size = [10] # array of hidden dimensions for the hidden layer for fully-connected NN or None-Type
seq_division = 2 # preprocessing tuple size e.g. [h1,h2,h3] --> [(h1,h2),(h2,h3),(h3,h1)]
# training procedure
optimizer = optim.SGD
opt_params = (1e-3,) # learning rate, other parameters left as default
N_epochs = 15
num_reps = 5
batch_size = 128 # batch size per device
model_name_scheme = "data/models/{}_{}_{}.pt"
rnn_load_name = "data/models/feats_small_{}_{}.pt"
transfer_load_name = "data/models/feats_init_{}_{}.pt"
scratch_load_name = "data/models/feats_scratch_{}_{}.pt"
# {full, small, switched, ...} x {rnn, model} x {num_run}

retrain_model = False # checks for availability of existing model file first if left at False.
# Otherwise, retrains entirely

N_train_filename = "data/mse_N/{}_loss_L_{}_N_{}.txt"
N_trains_plot = np.unique(np.round(np.logspace(0,2,19),0).astype(int))
retrain_mse_N = False # setting it to True takes quite some run time

mute_outputs = False  # verbose level during training
################################################################################################################
# save figures? Provide a folder for the images or set to None-Type for no saving of plots
save_figures = "plots/"
################################################################################################################

def filter_data(data,vals,low=2,high=10,include=True):
    """ Only includes data corresponding to vals from an interval [low,high].
        If include is set to False, exclude data from this interval, instead.
    """
    assert low <= high, "Lower boundary should be chosen smaller or equal to upper boundary."
    assert len(data) == len(vals), "Data was not of equal length"
    assert len(vals.shape) == 1, "Can only process 1-d array for vals"
    keep = np.bitwise_and(vals >= low , vals <= high)
    if not include:
        keep = np.bitwise_not(keep)
    return data[keep], vals[keep]

################################################################################################################
# Section 0 - initial data loading
################################################################################################################
neural_net = Approximator(L,energies=eps,add_energy_density=use_energy_feature,eps_shift=energy_shift)
# data loading and model init
neural_net.set_up(fileloc,
                  rnn_params=(feature_size,depth),
                  model_params=hidden_size,
                  seq_division=seq_division,
                  use_LSTM=use_LSTM,
                  learn_single_ind=single_inds,
                  N_train=N_train)
# resets the model parameters but can also be used for altering the model architecture
neural_net.reset_model()
# regardless of training, set the optimizer and its parameters
neural_net.optimizer = optimizer
neural_net.opt_params = opt_params

eps_idx = np.argmin(eps==0.5)

print(neural_net.model)
print(neural_net.rnn_kernel)

################################################################################################################
# Section 1 - single model
################################################################################################################
if show_single_model:
    print()
    for i in range(num_reps):
        neural_net.reset_model()
        model_name = model_name_scheme.format("small","{}",i)
        if use_energy_feature:
            # preload trained RNN part
            try:
                neural_net.load(rnn_load_name.format("{}",i),only_rnn=True)
            except:
                assert False, "Pretrained RNN-part could not be loaded from file."
            rnn = neural_net.rnn_kernel.state_dict()
            neural_net.reset_model()
            neural_net.rnn_kernel.load_state_dict(rnn)
        if retrain_model:
            print("Retraining model from scratch ...")
            neural_net.Lvals = L_small
            if use_energy_feature:
                neural_net.fix_features("train")
                neural_net.train_from_features(
                                          epochs=N_epochs,
                                          batch_size=batch_size*NUM_DEVICES,
                                          mute_outputs=mute_outputs
                                         )
            else:
                neural_net.train(epochs=N_epochs,batch_size=batch_size*NUM_DEVICES,mute_outputs=mute_outputs)
            print("Retraining finished, saving to file.")
            neural_net.save(model_name)
        else:
            print("Trying to load model from file ...")
            try:
                neural_net.load(model_name)
            except:
                print("File not found, training the model ...")
                neural_net.Lvals = L_small
                if use_energy_feature:
                    neural_net.fix_features("train")
                    neural_net.train_from_features(
                                          epochs=N_epochs,
                                          batch_size=batch_size*NUM_DEVICES,
                                          mute_outputs=mute_outputs
                                         )
                else:
                    neural_net.train(epochs=N_epochs,batch_size=batch_size*NUM_DEVICES,mute_outputs=mute_outputs)
                print("Training finished, saving to file.")
                neural_net.save(model_name)
            else:
                print("Model loaded from file.")
        # reset L
        neural_net.Lvals = L

        # print the number of parameters in the RNN and the consecutive model
        neural_net.num_params()

        # create fig 3 in paper
        temp = None if save_figures is None else save_figures + "indicator_plot_{}" + "_{}".format(i)
        plot.plot_inds(neural_net,label_idxs=single_inds,eps_idx=eps_idx,savename=temp)

        # create fig 8 from appendix
        if show_features:
            temp = None if save_figures is None else save_figures + "features_{}".format(i)
            plot.plot_features_all(neural_net,key="train",savename=temp)

################################################################################################################
# Section 2 - Coefficient of determination
################################################################################################################

if show_coefficient:
    model_list = [None]*num_reps
    for model_idx in range(num_reps):
        neural_net.reset_model()
        # checks whether to (re)train the model or to load from file on smaller training data
        model_name = model_name_scheme.format("small","{}",model_idx)
        if retrain_model and not show_single_model:
            # if show_single_model is True, model is already trained and saved to file
            neural_net.Lvals = L_small # for training purposes
            print("Retraining model from scratch ...")
            if use_energy_feature:
                neural_net.reset_model()
                neural_net.load(rnn_load_name.format("{}",model_idx),only_rnn=True)
                # get features based on the RNN loaded
                neural_net.fix_features("train")
                neural_net.train_from_features(
                                      epochs=N_epochs,
                                      batch_size=batch_size*NUM_DEVICES,
                                      mute_outputs=mute_outputs
                                     )
            else:
                neural_net.train(epochs=N_epochs,batch_size=batch_size*NUM_DEVICES,mute_outputs=mute_outputs)
            print("Retraining finished, saving to file.")
            neural_net.save(model_name)
            neural_net.Lvals = L # for full test set analysis
        else:
            print("Trying to load model from file ...")
            try:
                neural_net.load(model_name)
            except:
                neural_net.Lvals = L_small # for training purposes
                print("File not found, training the model ...")
                neural_net.reset_model()
                if use_energy_feature:
                    neural_net.load(rnn_load_name.format("{}",model_idx),only_rnn=True)
                    neural_net.fix_features("train")
                    neural_net.train_from_features(
                                          epochs=N_epochs,
                                          batch_size=batch_size*NUM_DEVICES,
                                          mute_outputs=mute_outputs
                                         )
                else:
                    neural_net.train(epochs=N_epochs,batch_size=batch_size*NUM_DEVICES,mute_outputs=mute_outputs)
                print("Training finished, saving to file.")
                neural_net.save(model_name)
                neural_net.Lvals = L # for full test set analysis
            else:
                print("Model loaded from file.")
        model_list[model_idx] = model_name
    # coefficient of determination (fig 4 in paper)
    temp = None if save_figures is None else save_figures + "coefficient_of_determination_eps_{}"
    for e in eps_show:
        eps_idx = np.argmax(eps==e)
        plot.plot_r2(neural_net,model_list,label_idxs=single_inds,eps_idx=eps_idx,savename=temp.format(e))

################################################################################################################
# Section 3 - Transfer learning
################################################################################################################
if show_transfer:
    # transfer learning: Train on two indicators first, then drop the post-processing model
    # Finally, retrain on the third indicator
    multitask, transfer, scratch = [], [], []
    
    neural_net = Approximator(L,energies=eps,add_energy_density=use_energy_feature,eps_shift=energy_shift)
    # data loading and model init
    neural_net.set_up(fileloc,
                      rnn_params=(feature_size,depth),
                      model_params=hidden_size,
                      seq_division=seq_division,
                      use_LSTM=use_LSTM,
                      learn_single_ind=single_inds,
                      N_train=N_train)
    neural_net.optimizer = optimizer
    neural_net.opt_params = opt_params

    transfer_net = Approximator(L,energies=eps,add_energy_density=use_energy_feature,eps_shift=energy_shift)
    # data loading and model init
    transfer_net.set_up(fileloc,
                      rnn_params=(feature_size,depth),
                      model_params=hidden_size,
                      seq_division=seq_division,
                      use_LSTM=use_LSTM,
                      learn_single_ind=single_inds,
                      N_train=N_train)
    transfer_net.optimizer = optimizer
    transfer_net.opt_params = opt_params

    adversary_net = Approximator(L,energies=eps,add_energy_density=use_energy_feature,eps_shift=energy_shift)
    # data loading and model init
    adversary_net.set_up(fileloc,
                      rnn_params=(feature_size,depth),
                      model_params=hidden_size,
                      seq_division=seq_division,
                      use_LSTM=use_LSTM,
                      learn_single_ind=[single_inds[t] for t in transfer_inds],
                      N_train=N_train)
    adversary_net.optimizer = optimizer
    adversary_net.opt_params = opt_params
    
    for model_idx in range(num_reps):
        for net,listing,label,load_name in zip([neural_net,adversary_net],[multitask,scratch],
                                             ["small","transfer_adversary"],(rnn_load_name,scratch_load_name)):
            model_name = model_name_scheme.format(label,"{}",model_idx)
            net.reset_model()
            if retrain_model and not (show_single_model and label=="small"):
                net.Lvals = L_small # for training purposes
                print("Retraining model from scratch ...")
                if use_energy_feature:
                    net.load(load_name.format("{}",model_idx),only_rnn=True)
                    # get features based on the RNN loaded
                    net.fix_features("train")
                    net.train_from_features(
                                      epochs=N_epochs,
                                      batch_size=batch_size*NUM_DEVICES,
                                      mute_outputs=mute_outputs
                                     )
                else:
                    net.train(epochs=N_epochs,batch_size=batch_size*NUM_DEVICES,mute_outputs=mute_outputs)
                print("Retraining finished, saving to file.")
                net.save(model_name)
                net.Lvals = L # for full test set analysis
            else:
                print("Trying to load model from file ...")
                try:
                    net.load(model_name)
                except:
                    net.Lvals = L_small # for training purposes
                    print("File not found, training the model ...")
                    if use_energy_feature:
                        net.load(load_name.format("{}",model_idx),only_rnn=True)
                        # get features based on the RNN loaded
                        net.fix_features("train")
                        net.train_from_features(
                                      epochs=N_epochs,
                                      batch_size=batch_size*NUM_DEVICES,
                                      mute_outputs=mute_outputs
                                     )
                    else:
                        net.train(epochs=N_epochs,batch_size=batch_size*NUM_DEVICES,mute_outputs=mute_outputs)
                    print("Training finished, saving to file.")
                    net.save(model_name)
                    net.Lvals = L # for full test set analysis
                else:
                    print("Model loaded from file.")
            listing.append(model_name)
            
        transfer_net.reset_model()
        transfer_net.opt_params = opt_params
        if use_energy_feature:
            transfer_net.load(transfer_load_name.format("{}",model_idx),only_rnn=True)
        model_name = model_name_scheme.format("transfer_init","{}",model_idx)
        if retrain_model:
            transfer_net.Lvals = L_small # for training purposes
            print("Retraining model from scratch ...")
            if use_energy_feature:
                transfer_net.train_transfer(init_inds,
                                      epochs=N_epochs,
                                      batch_size=batch_size*NUM_DEVICES,
                                      mute_outputs=mute_outputs
                                     )
            else:
                transfer_net.train(epochs=N_epochs,
                                 batch_size=batch_size*NUM_DEVICES,
                                 mute_outputs=mute_outputs,
                                 single_inds=init_inds
                                )
            print("Retraining finished, saving to file.")
            transfer_net.save(model_name)
            transfer_net.Lvals = L # for full test set analysis
        else:
            print("Trying to load model from file ...")
            N_inds = transfer_net.N_inds
            try:
                # juggle with the number of indicators for loading procedure
                transfer_net.N_inds = len(init_inds)
                transfer_net.reset_model()
                transfer_net.load(model_name)
            except:
                transfer_net.N_inds = N_inds
                transfer_net.Lvals = L_small # for training purposes
                print("File not found, training the model ...")
                if use_energy_feature:
                    transfer_net.train_transfer(init_inds,
                                      epochs=N_epochs,
                                      batch_size=batch_size*NUM_DEVICES,
                                      mute_outputs=mute_outputs
                                     )
                else:
                    transfer_net.train(epochs=N_epochs,
                                     batch_size=batch_size*NUM_DEVICES,
                                     mute_outputs=mute_outputs,
                                     single_inds=init_inds
                                    )
                print("Training finished, saving to file.")
                transfer_net.save(model_name)
                transfer_net.Lvals = L # for full test set analysis
            else:
                transfer_net.N_inds = N_inds
                print("Model loaded from file.")
        # check whether to retrain or load from file for transfer training
        model_name = model_name_scheme.format("transfer","{}",model_idx)
        transfer_net.opt_params = (5e-5,)
        if retrain_model:
            transfer_net.Lvals = L_small # for training purposes
            print("Retraining model from scratch ...")
            transfer_net.train_transfer(ind_idxs=transfer_inds,
                             epochs=40,
                             batch_size=batch_size*NUM_DEVICES,
                             mute_outputs=mute_outputs
                            )
            print("Retraining finished, saving to file.")
            transfer_net.save(model_name)
            transfer_net.Lvals = L # for full test set analysis
        else:
            print("Trying to load model from file ...")
            N_inds = transfer_net.N_inds
            try:
                # juggle with the number of indicators for loading procedure
                transfer_net.N_inds = len(transfer_inds)
                transfer_net.reset_model()
                transfer_net.load(model_name)
            except:
                transfer_net.N_inds = N_inds
                transfer_net.Lvals = L_small # for training purposes
                print("File not found, training the model ...")
                transfer_net.train_transfer(ind_idxs=transfer_inds,
                                 epochs=40,
                                 batch_size=batch_size*NUM_DEVICES,
                                 mute_outputs=mute_outputs
                                )
                print("Training finished, saving to file.")
                transfer_net.save(model_name)
                transfer_net.Lvals = L # for full test set analysis
            else:
                transfer_net.N_inds = N_inds
                print("Model loaded from file.")
                
        transfer.append(model_name)

    model_names = np.array([transfer,multitask,scratch]).T

    # create fig 5 in paper concerning transfer learning
    temp = None if save_figures is None else save_figures + "transfer_learning_eps_{}_feat_{}"
    for e in eps_show:
        eps_idx = np.argmax(eps==e)
        plot.plot_r2_comparison(transfer_net,neural_net,adversary_net,transfer_inds,model_names,
                                eps_idx=eps_idx,savename=temp.format(e,feature_size))

################################################################################################################
# Section 4 - Phase diagram
################################################################################################################
if show_phase_diag:
    L_small = L
    eps_full = np.arange(1,20)/20
    neural_net = Approximator(L,energies=eps_full,add_energy_density=False,eps_shift=energy_shift)
    # data loading and model init
    neural_net.set_up(fileloc,
                      rnn_params=(2,depth),
                      model_params=[10],
                      seq_division=seq_division,
                      use_LSTM=use_LSTM,
                      learn_single_ind=single_inds,
                      N_train=N_train)
    # resets the model parameters but can also be used for altering the model architecture
    neural_net.reset_model()
    # regardless of training, set the optimizer and its parameters
    neural_net.optimizer = optim.Adam
    neural_net.opt_params = (1e-3,)
    model_name = model_name_scheme.format("phase","diag","{}")

    if retrain_model:
        # first training stage - train all energy sectors individually
        print("Retraining model from scratch ...")
        neural_net.Lvals = L_small
        neural_net.train(5,batch_size*NUM_DEVICES)

        # second training stage - keep rnn part fixed and retrain NN based on extracted features and energy density
        rnn = neural_net.rnn_kernel.state_dict()
        neural_net.opt_params = (1e-4,)
        neural_net.add_energy_density = True
        neural_net.fix_features("train")
        neural_net.reset_model(model_params=[31]) # hidden size of NN
        neural_net.rnn_kernel.load_state_dict(rnn)
        neural_net.train_from_features(epochs=5,batch_size=batch_size*NUM_DEVICES)
        print("Retraining finished, saving to file.")
        neural_net.save(model_name)
        neural_net.Lvals = L
    else:
        print("Trying to load model from file ...")
        
        try:
            neural_net.add_energy_density = True
            neural_net.reset_model(model_params=[31]) # hidden size of NN
            neural_net.load(model_name)
        except:
            neural_net.Lvals = L_small # for training purposes
            print("File not found, training the model ...")
            neural_net.train(5,batch_size*NUM_DEVICES)
            rnn = neural_net.rnn_kernel.state_dict()
            neural_net.opt_params = (1e-4,)
            neural_net.add_energy_density = True
            neural_net.fix_features("train")
            neural_net.reset_model(model_params=[31]) # hidden size of NN
            neural_net.rnn_kernel.load_state_dict(rnn)
            neural_net.train_from_features(epochs=5,batch_size=batch_size*NUM_DEVICES)
            print("Training finished, saving to file.")
            neural_net.save(model_name)
            neural_net.Lvals = L
        else:
            print("Model loaded from file.")
    
    # plot phase diagram based on the previous training (fig 6)
    plot_info = { "numberpos": (0.9,0.15) } # possible alterations to design of plots
    temp = None if save_figures is None else save_figures + "phase_diag_L_14"
    plot.plot_phase_diagram(neural_net,14,8,h_max=11,savename=temp,plot_info=plot_info)

################################################################################################################
# Appendix - N_train loss curve
################################################################################################################

N_max = 100
N_epochs = 30

if retrain_mse_N:
    # probe between N_train = 1, ... , 100
    N_new = N_trains_plot
    # use data bounds?
    bounds = None
    
    N_epochs_RNN = 15
    optimizer_RNN = optim.SGD
    opt_params_RNN = (1.0535e-5,)

    const_RNN = N_max*N_epochs_RNN
    const     = N_max*N_epochs
    for N in N_new:
        N_e_RNN = int(np.ceil(const_RNN/N))
        N_e     = int(np.ceil(const/N))
        print(N,"\t",N_e*N,"\t",N_e_RNN*N)

    const_RNN = N_max*N_epochs_RNN
    const     = N_max*N_epochs

    losses = []            

    for N in reversed(N_new):
        # adjust number of total epochs
        N_e_RNN = int(np.ceil(const_RNN/N))
        N_e     = int(np.ceil(const/N))

        temp_losses = []

        for n in range(num_reps):
            neural_net = Approximator(L,energies=eps,add_energy_density=False,eps_shift=energy_shift)
            # data loading and model init
            neural_net.set_up(fileloc,
                              rnn_params=(feature_size,depth),
                              model_params=hidden_size,
                              seq_division=seq_division,
                              use_LSTM=use_LSTM,
                              learn_single_ind=single_inds,
                              N_train=N)
            # resets the model parameters but can also be used for altering the model architecture
            neural_net.reset_model()
            # regardless of training, set the optimizer and its parameters
            neural_net.optimizer = optimizer_RNN
            neural_net.opt_params = opt_params_RNN

            print(neural_net.model)
            print(neural_net.rnn_kernel)

            # train RNN first
            if bounds is not None:
                # cut down on the values for h for the available training data but keep the initial data
                key_data = "train"
                data_copy = []
                for l in neural_net.Lvals:
                    data = neural_net.data[l][key_data]
                    data_copy.append(data.copy())
                    for key in ["h_i","inds"]:
                        data[key], temp = filter_data(data[key],data["hcorr"],*bounds)
                    data["h"] = np.sort(np.unique(temp))
                    data["hcorr"] = temp

            if use_energy_feature:
                neural_net.Lvals = L_small
                neural_net.train(epochs=N_e_RNN,
                             batch_size=batch_size*NUM_DEVICES,
                             mute_outputs=mute_outputs)
                neural_net.save("temp_{}.pt")
                neural_net.Lvals = L

            # restore data in case it was cut down
            if bounds is not None:
                for data,l in zip(data_copy,neural_net.Lvals):
                    neural_net.data[l][key_data] = data

            # keep the RNN for the subsequent training fixed
            neural_net.add_energy_density = use_energy_feature
            neural_net.reset_model(model_params=hidden_size)
            neural_net.optimizer = optimizer
            neural_net.opt_params = opt_params

            temp = 2
            while temp > 1:
                # retry if training does not succeed which is visible in a high loss
                # retrain subsequent model after RNN
                print("Retraining model from scratch ...")
                neural_net.reset_model()
                neural_net.Lvals = L_small
                if use_energy_feature:
                    neural_net.load("temp_{}.pt",only_rnn=True)
                    # get features based on the RNN loaded
                    neural_net.fix_features("train")
                    neural_net.fix_features("test")
                    neural_net.Lvals = L_small # for training purposes
                    _, loss_curve = neural_net.train_from_features(epochs=N_e,
                                                                   batch_size=batch_size*NUM_DEVICES,
                                                                   mute_outputs=mute_outputs,
                                                                   L_tests=L
                                                                  )
                else:
                    _, loss_curve = neural_net.train(epochs=N_e,batch_size=batch_size*NUM_DEVICES,mute_outputs=mute_outputs,L_tests=L)
                neural_net.Lvals = L # for full test set analysis
                temp = np.mean(loss_curve[0,-1]) # retry if final loss is above threshold
            temp_losses.append(loss_curve)

        # data saving
        losses.append(temp_losses)

    # save all losses
    for j,l in enumerate(L):
        for k,key in enumerate(["train","test"]):
            for loss,N in zip(losses,reversed(N_new)):
                np.savetxt(N_train_filename.format(key,l,N),np.array(loss)[:,k,:,j].T)

if show_N_train_curve:
    temp = None if save_figures is None else save_figures + "N_dep"
    prefactor = 30*len(L_small)/batch_size
    # fig 9
    plot.plot_N_train_dependency(N_trains_plot,L,prefactor,N_train_filename,N_max=N_max,N_epochs=N_epochs,savename=temp,num_mins=10)


