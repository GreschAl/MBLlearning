##################################################################################################
# Convenience functions to create plots
# Usage is completeley optional
##################################################################################################

import numpy as np
from torch import from_numpy, full
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True })
plt.rc('font', family='serif')
plt.style.use('tableau-colorblind10')

from MBLlearning.utils.data import get_avg, coefficient_of_determination
from MBLlearning.learn_inds.recurrent import get_LSTM_data
from MBLlearning.global_config import diction

##################################################################################################

def plot_features_all(net,key="test",show_std=False,savename=None,show_plot=False):
    """ Plots the feature(s) produced by the RNN for the available data at various chain lengths at once.
        If show_std set to True (not by default) displays the single standard deviation, otherwise the error on the mean.
        If savename is provided, saves the images to file.
    """
    features = []
    for l in net.Lvals:
        h, hcorr, feat = net.get_features(l,key)
        temp = []
        for hc in h:
            keep = hcorr == hc
            temp.append(feat[keep])
        features.append(np.array(temp))
    features = np.array(features)
    means, stds = np.mean(features,axis=-2), np.std(features,axis=-2)/np.sqrt(features.shape[-2])
    
    N = means.shape[-1]
    assert N < 10, "Cannot plot features for a feature size of 10 or above."
    #plt.figure(figsize=(6,8))
    for n in range(N):
        # go over each feature
        plt.subplot(100*(N//2)+20+1+n)
        for mean,std,l in zip(means,stds,net.Lvals):
            if l == 14:
                plt.plot([],[])
            if show_std:
                plt.errorbar(h,mean[...,n],yerr=std[...,n],label="$L = {}$".format(l))
            else:
                plt.plot(h,mean[...,n],label="$L = {}$".format(l))
        if n == 1:
            plt.legend(fontsize="x-large")
        plt.ylabel("Feature value",fontsize="x-large")
        if n%2==1:
            # set right yticks to right side
            plt.gca().yaxis.set_label_position("right")
            plt.gca().yaxis.set_ticks_position("right")
        plt.yticks(fontsize="x-large")
        plt.xticks(fontsize="x-large")
        plt.xlabel("Disorder parameter $h$",fontsize="x-large")
    plt.subplots_adjust(wspace=0.05)
    if savename is not None:
        plt.savefig(savename+"_mean.pdf",orientation="landscape",dpi=600,bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()
    return

def plot_inds(net,label_idxs=[0,1,2],eps_idx=9,savename=None,L_vals=None,show_plot=False):
    for L in net.Lvals:
        h, singlescores, targets = net.predict_energy_wise(L)
        
        # average scores over disorder realizations
        scores = np.mean(singlescores,axis=2)
        std = np.std(singlescores,axis=2)

        h, means, stds, _ = get_avg(net,L)

        plt.figure(figsize=(8,6))
        use_even = np.arange(len(h))%2==0
        use_odd  = np.bitwise_not(use_even)
        for n,idx in enumerate(label_idxs):
            i = [2,0,1][n]
            ax = plt.subplot(221+i)
            plt.errorbar(h[use_even],scores[eps_idx,use_even,n],
                         yerr=std[eps_idx,use_even,n],marker=".",linewidth=0,elinewidth=2,label="Estimation")
            plt.errorbar(h[use_odd],means[eps_idx,use_odd,n],
                         yerr=stds[eps_idx,use_odd,n],marker=".",linewidth=0,elinewidth=2,label="Exact diagonalization")        
            if i!=0:
                plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
            plt.xticks(np.arange(2,16,2),fontsize="x-large")
            #plt.text(*numberpos[i],chr(ord('a')+i)+")",fontsize=22)
            plt.grid()
            plt.xlim(1,15)
            #plt.ylim(ylims_dict[i])
            if i%2==1:
                # set right yticks to right side
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.set_ticks_position("right")
            plt.yticks(fontsize="x-large")
            plt.ylabel(diction[idx],fontsize="xx-large")
            if i==2:
                plt.legend(bbox_to_anchor=[2.1,0.65],fontsize="x-large")
        
        plt.subplots_adjust(wspace=0.05,hspace=0.2)
        if savename is not None:
            plt.savefig(savename.format(L)+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close()
    return

def plot_r2(net,model_list=None,label_idxs=[0,1,2],eps_idx=9,savename=None,show_plot=False):
    
    calc_averages = model_list is not None
    
    plt.figure(figsize=(8,6))
    for n,idx in enumerate(label_idxs):
        i = [2,0,1][n]
        if not calc_averages:
            ax = plt.subplot(221+i)
            for l in net.Lvals:
                h = net.data[l]["test"]["h"]
                r_squared = coefficient_of_determination(net,l,reduce_energies=True,key="test")
                r_squared = r_squared[eps_idx]
                r_normed  = 1/(2-r_squared)
                if l!=14:
                    plt.plot(h,r_normed[:,n],label="L={}".format(l))
                else:
                    plt.plot([],[])
                    plt.plot(h,r_normed[:,n],linestyle="dashed",label="L={}".format(l))

            plt.hlines(0.5,0,16,colors="black",linestyles="dotted")
            if i!=0:
                plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
            plt.xticks(np.arange(1,16,2),fontsize="x-large")
            if i%2==0:
                # set right yticks to right side
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.set_ticks_position("right")
                plt.yticks(fontsize="x-large")
            else:
                plt.yticks(np.arange(0,11,2)/10,[])
            if i==2:
                plt.ylabel("$R^2_{norm.}$",fontsize="x-large")
            plt.ylim(0,1)
            plt.xlim(0,15.5)
            plt.text(0.65,0.25,diction[idx],fontsize=18,horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,bbox=dict(facecolor='white', alpha=0.5))
            if i==2:
                plt.legend(bbox_to_anchor=[2,0.75],fontsize="xx-large")
        else:
            ax = plt.subplot(221+i)
            for l in net.Lvals:
                r_norms = []
                for model_name in model_list:
                    net.load(model_name)
                    net.clear_estimates()
                    h = net.data[l]["test"]["h"]
                    r_squared = coefficient_of_determination(net,l,reduce_energies=True,key="test")
                    r_squared = r_squared[eps_idx]
                    r_normed  = 1/(2-r_squared)
                    r_norms.append(r_normed[:,n])
                if l!=14:
                    plt.plot(h,np.mean(r_norms,axis=0),label="L={}".format(l))
                else:
                    plt.plot([],[])
                    plt.plot(h,np.mean(r_norms,axis=0),linestyle="dashed",label="L={}".format(l))

            plt.hlines(0.5,0,16,colors="black",linestyles="dotted")
            if i!=0:
                plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
            plt.xticks(np.arange(1,16,2),fontsize="x-large")
            if i%2==0:
                # set right yticks to right side
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.set_ticks_position("right")
                plt.yticks(fontsize="x-large")
            else:
                plt.yticks(np.arange(0,11,2)/10,[])
            if i==2:
                plt.ylabel("$R^2_{norm.}$",fontsize="x-large")
            plt.ylim(0,1)
            plt.xlim(0,15.5)
            plt.text(0.65,0.25,diction[idx],fontsize=18,horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,bbox=dict(facecolor='white', alpha=0.5))
            if i==2:
                plt.legend(bbox_to_anchor=[2,0.75],fontsize="xx-large")
    plt.subplots_adjust(wspace=0.15,hspace=0.2)
    if savename is not None:
        plt.savefig(savename+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()
    return

def plot_N_train_dependency(N_vals,L,prefactor,load_name,N_max=1000,N_epochs=5,num_mins=1,savename=None,show_plot=False):
    
    # load data
    losses = np.empty((len(L),2,len(N_vals)),dtype=object)
    L_are_available = np.zeros(len(L),dtype=bool)
    for j,l in enumerate(L):
        for k,key in enumerate(["train","test"]):
            for i,N in enumerate(N_vals):
                try:
                    losses[j,k,i] = np.loadtxt(load_name.format(key,l,N))
                    L_are_available[j] = True
                except:
                    losses[j,k,i] = None
                    print("No data for (L,key,N_train) = ({},{},{}) found on file.".format(l,key,N))
                    print("Skipping plotting. Please provide data first via training")
    
    # process data
    mses = [ [], [] ]
    for i,(l,temp) in enumerate(zip(L,losses)):
        # plot up to the fouth available chain lengths
        mse = []
        for k,(key,loss) in enumerate(zip(["train","test"],temp)):
            mse = []
            for j,N in enumerate(N_vals):
                N_e = int(np.ceil(N_max*N_epochs/N))
                num_updates = np.arange(1,N_e+1)*np.ceil(N*prefactor)
                keep = np.bitwise_and(num_updates >= 1300,num_updates<=1400)
                means = np.sort(np.median(loss[j][keep],axis=0))
                mse.append(np.mean(means[:num_mins]))
            mses[k].append(np.array(mse))
    mses = np.array(mses)

    # plot data
    marker = ["o","s","X"]
    for s,(key,mse_key) in enumerate(zip(["train","test"],mses)):
        if s == 1:
            plt.gca().set_prop_cycle(None)
        for m,l,mse in zip(marker,L,mse_key):
            label = "$L={}$:".format(l) if s==0 else "train - test"
            if l == 14:
                plt.plot([],[])
            if s==0:
                plt.semilogx(N_vals,mse,marker=m,linewidth=0,label=label)
            else:
                plt.semilogx(N_vals,mse,marker=m,markerfacecolor='none',linewidth=0,label=label)
    plt.legend(ncol=2,loc="upper right",fontsize="x-large",
               borderpad=0.4,handletextpad=0.4,columnspacing=0.0,markerfirst=False)
    plt.ylim(2e-3,8e-3)
    plt.ylabel("Avg. MSE per sample $\\left[\\times 10^{-3} \\right]$",fontsize="x-large")
    plt.yticks(np.arange(2,9)/1000,np.arange(2,9),fontsize="x-large")
    plt.xticks([1,3,10,100],[1,3,10,100],fontsize="x-large")
    plt.xlabel("$N_{train}$",fontsize="xx-large")
    plt.vlines(2.5,1e-3,9e-3,colors="black",linestyles="dotted",linewidth=1)
    plt.grid(axis="y")
    
    if savename is not None:
        plt.savefig(savename+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
    if show_plot:   
        plt.show()
    else:
        plt.close()
    return

def plot_r2_comparison(transfer,net,adversary,transfer_idxs,model_names=None,eps_idx=9,savename=None,show_plot=False):
    """ The arguments (transfer,net,adversary) can either be Approximator-Class of lists thereof.
        In the latter case, calculates means and standard deviations of the outputs
    """
    calc_averages = model_names is not None
    
    plt.figure(figsize=(8,6))
    for idx in transfer_idxs:
        if calc_averages:
            for i,l in enumerate(net.Lvals):
                plt.subplot(221+i)
                ### evaluate baseline neural network trained on all indicators
                h = net.data[l]["test"]["h"]
                r_normed = []
                for net_name in model_names:
                    net.load(net_name[1])
                    net.clear_estimates()
                    r_squared = coefficient_of_determination(net,l,reduce_energies=True,key="test")
                    r_squared = r_squared[eps_idx,:,idx]
                    r_normed.append( 1/(2-r_squared) )
                if l!=14:
                    plt.plot(h,np.mean(r_normed,axis=0),label="multitask")
                else:
                    plt.plot(h,np.mean(r_normed,axis=0),linestyle="dashed")

                ### evaluate adversarial neural network trained on only the transfer indicator
                h = adversary.data[l]["test"]["h"]
                r_normed = []
                for adv_name in model_names:
                    adversary.load(adv_name[2])
                    adversary.clear_estimates()
                    r_squared = coefficient_of_determination(adversary,l,reduce_energies=True,key="test")
                    r_squared = r_squared[eps_idx]
                    temp = 1/(2-r_squared)
                    r_normed.append( temp[:,0] )
                if l!=14:
                    plt.plot(h,np.mean(r_normed,axis=0),label="scratch")
                else:
                    plt.plot(h,np.mean(r_normed,axis=0),linestyle="dashed")

                ### evaluate transfer neural network trained in two phases
                plt.plot([],[]) # for right color
                
                h = transfer.data[l]["test"]["h"]
                r_normed = []
                for transfer_name in model_names:
                    transfer.load(transfer_name[0])
                    transfer.clear_estimates()
                    r_squared = coefficient_of_determination(transfer,l,reduce_energies=True,
                                                                        key="test",transfer_ind=idx)
                    r_squared = r_squared[:,eps_idx]
                    r_normed.append( 1/(2-r_squared) )
                
                if l!=14:
                    plt.plot(h,np.mean(r_normed,axis=0),label="transfer")
                else:
                    plt.plot(h,np.mean(r_normed,axis=0),linestyle="dashed")
                
                plt.hlines(0.5,0,16,colors="black",linestyles="dotted")
                if i>0:
                    plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
                plt.xticks(np.arange(1,16,2),fontsize="x-large")
                if i%2==0:
                    # set right yticks to right side
                    plt.gca().yaxis.set_label_position("right")
                    plt.gca().yaxis.set_ticks_position("right")
                    plt.yticks(fontsize="x-large")
                else:
                    plt.yticks(np.arange(0,11,2)/10,[])
                if i==2:
                    plt.ylabel("$R^2_{norm.}$",fontsize="x-large")
                plt.ylim(0,1)
                plt.xlim(0,15.5)
                plt.text(10,0.85,"$L={}$".format(l),fontsize="xx-large")
                if i==1:
                    plt.legend(bbox_to_anchor=[0.85,-0.45],fontsize="xx-large")
        else:
            for i,l in enumerate(net.Lvals):
                plt.subplot(221+i)
                ### evaluate baseline neural network trained on all indicators
                h = net.data[l]["test"]["h"]
                r_squared = coefficient_of_determination(net,l,reduce_energies=True,key="test")
                r_squared = r_squared[eps_idx,:,idx]
                r_normed  = 1/(2-r_squared)
                if l!=14:
                    plt.plot(h,r_normed,label="multitask")
                else:
                    plt.plot(h,r_normed,linestyle="dashed")

                ### evaluate adversarial neural network trained on only the transfer indicator
                h = adversary.data[l]["test"]["h"]
                r_squared = coefficient_of_determination(adversary,l,reduce_energies=True,key="test")
                r_squared = r_squared[eps_idx]
                r_normed  = 1/(2-r_squared)
                if l!=14:
                    plt.plot(h,r_normed[:,0],label="scratch")
                else:
                    plt.plot(h,r_normed[:,0],linestyle="dashed")

                ### evaluate transfer neural network trained in two phases
                plt.plot([],[]) # for right color
                
                h = transfer.data[l]["test"]["h"]
                r_squared = coefficient_of_determination(transfer,l,reduce_energies=True,
                                                                    key="test",transfer_ind=idx)

                r_squared = r_squared[:,eps_idx]
                r_normed  = 1/(2-r_squared)
                if l!=14:
                    plt.plot(h,r_normed,label="transfer")
                else:
                    plt.plot(h,r_normed,linestyle="dashed")

                plt.hlines(0.5,0,16,colors="black",linestyles="dotted")
                if i>0:
                    plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
                plt.xticks(np.arange(1,16,2),fontsize="x-large")
                if i%2==0:
                    # set right yticks to right side
                    plt.gca().yaxis.set_label_position("right")
                    plt.gca().yaxis.set_ticks_position("right")
                    plt.yticks(fontsize="x-large")
                else:
                    plt.yticks(np.arange(0,11,2)/10,[])
                if i==2:
                    plt.ylabel("$R^2_{norm.}$",fontsize="x-large")
                plt.ylim(0,1)
                plt.xlim(0,15.5)
                plt.text(10,0.85,"$L={}$".format(l),fontsize="xx-large")
                if i==1:
                    plt.legend(bbox_to_anchor=[0.85,-0.45],fontsize="xx-large")
    plt.subplots_adjust(wspace=0.15,hspace=0.2)
    if savename is not None:
        plt.savefig(savename+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()
    return

def increase_precision(arr,scale_factor,base_scale=2):
    """ Helper function for creating the phase diagram. Takes a sorted array with a base resolution of base-scale.
        Returns an array of resolution scale_factor*base_scale with the same start and end point of arr.
    """
    mins, maxs = np.min(arr), np.max(arr)
    assert arr[0] == mins and arr[-1] == maxs, "Array has to be sorted."
    scale_factor *= base_scale
    R = (len(arr)-1) / base_scale # range of array
    N = int(scale_factor*R + 1) # rescale
    return np.linspace(mins,maxs,N)

def get_phase_diagram_data(net,L_val,scale_factor,eps_train=None,N_samples=1000,h_max=30):
    assert net.add_energy_density, "Phase diagram only available for architecture with variable energy values."
    eps_test = increase_precision(net.energies,scale_factor)
    h = net.data[L_val]["train"]["h"]
    h = h[h <= h_max]
    h_test = increase_precision(h,scale_factor)
    temp = None if eps_train is None else {"eps": eps_train, "h": h}
    out = {"init": temp, "data": {"eps": net.energies, "h": h}, "net": {"eps": eps_test, "h": h_test}}
    out["data"]["scale_factor"] = scale_factor
    # mock data generation
    seq_division = net.rnn_params["seq_division"]
    data = []
    data_std = []
    for e in eps_test:
        eps_data = []
        eps_data_std = []
        for hval in h_test:
            # sample, reshape for RNN, then predict
            temp = np.random.uniform(-hval,hval,size=(N_samples,L_val))
            temp = from_numpy(get_LSTM_data(temp,seq_division=seq_division))
            temp = net._predict_data(temp,eps=full((N_samples,1),e)) # eps shift is applied later
            eps_data.append(np.mean(temp,axis=0))
            eps_data_std.append(np.std(temp,axis=0))
        data.append(np.array(eps_data))
        data_std.append(np.array(eps_data_std))
    data = np.array(data).T
    data_std = np.array(data_std).T # N_inds x len(h) x len(eps)
    out["net"]["data"] = data
    out["net"]["std"] = data_std
    
    # prepare train data from loaded model to compare
    source = net.data[L_val]["train"]
    inds = net.sort(source["inds"])
    data_comp = []
    data_comp_std = []
    for temp in inds:
        h_data = []
        h_data_std = []
        for hval in h:
            keep = source["hcorr"] == hval
            # keep only the right data for each and average afterwards
            h_data.append(np.mean(temp[keep],axis=0))
            h_data_std.append(np.std(temp[keep],axis=0))
        data_comp.append(np.array(h_data))
        data_comp_std.append(np.array(h_data_std))
    data_comp = np.array(data_comp).T
    data_comp_std = np.array(data_comp_std).T # N_inds x len(h) x len(eps)
    out["data"]["data"] = data_comp
    out["data"]["std"] = data_comp_std
    
    # if applicable, prepare data used for the training of the net
    if eps_train is not None:
        keeps = np.array([e in eps_train for e in net.energies])
        data_init = []
        data_init_std = []
        # keep only the corresponding inds that belong to eps_train
        for temp in inds[keeps]:
            h_data = []
            h_data_std = []
            for hval in h:
                keep = source["hcorr"] == hval
                # keep only the right data for each and average afterwards
                h_data.append(np.mean(temp[keep],axis=0))
                h_data_std.append(np.std(temp[keep],axis=0))
            data_init.append(np.array(h_data))
            data_init_std.append(np.array(h_data_std))
        data_init = np.array(data_init).T
        data_init_std = np.array(data_init_std).T # N_inds x len(h) x len(eps_train)
        out["init"]["data"] = data_init
        out["init"]["std"] = data_init_std

    return out
    
def plot_from_phase_diag_data(all_data,savename=None,show_plot=False,plot_info={}):
    aspect0 = plot_info.get("aspect",30/8.25)
    hspace  = plot_info.get("hspace",-0.3)
    numberpos = plot_info.get("numberpos",(0.5,0.5))
    
    scale_factor = all_data["data"]["scale_factor"]
    data, eps_vals, h_vals = [], [], []
    for d in all_data.values():
        if d is not None:
            data.append(d["data"])
            eps_vals.append(d["eps"])
            h_vals.append(d["h"])
        else:
            data.append(None)
            eps_vals.append(None)
            h_vals.append(None)
        
    if data[0] is not None:
        for i,diagrams in enumerate(zip(*data)):
            plt.figure(figsize=(8,8))
            for j,(diag,axes,scale,offset,axpos,aspect) in enumerate(zip(diagrams,
                                                        zip(h_vals,eps_vals),
                                                        (None,4,int(4*scale_factor)),
                                                        (None,1,int(scale_factor)),
                                                        ((0,1),(1,0),(1,2)),
                                                        (aspect0,"equal","equal")
                                                       )):
                x,y = diag.shape
                ax = plt.subplot2grid((2, 4), axpos, 1, 2)
                ax.imshow(diag.T,aspect=aspect,origin="lower")
                if scale is not None:
                    ax.set_xticks(np.arange(offset,x,scale))
                    ax.set_xticklabels(axes[0][offset::scale].astype(int),fontsize="x-large")
                    ax.set_yticks(np.arange(offset,y,scale))
                    ax.set_yticklabels(np.round(axes[1][offset::scale],1),fontsize="x-large")
                else:
                    ax.set_xticks(np.arange(1,x,4))
                    ax.set_xticklabels(axes[0][1::4].astype(int),fontsize="x-large")
                    ax.set_yticks(np.arange(0,y))
                    ax.set_yticklabels(np.round(axes[1],1),fontsize="x-large")
                ax.set_xlabel("Disorder parameter $h$",fontsize="x-large")
                ax.set_ylabel("Energy density $\\epsilon$",fontsize="x-large")
                ax.text(*numberpos,chr(ord('a')+j)+")",fontsize=24,color="white",horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
                if j==2:
                    # set right yticks to right side
                    plt.gca().yaxis.set_label_position("right")
                    plt.gca().yaxis.set_ticks_position("right")

            # adjust positioning
            plt.subplots_adjust(hspace=hspace)
            if savename is not None:
                plt.savefig(savename+"_{}.pdf".format(i),orientation="landscape",dpi=600,bbox_inches="tight")
            if show_plot:   
                plt.show()
            else:
                plt.close()
    else:
        # plot data if eps_train is not provided
        h_vals, eps_vals = h_vals[1:],eps_vals[1:]
        for i,diagrams in enumerate(zip(*data[1:])):
            plt.figure(figsize=(8,6))
            for j,(diag,axes,scale,offset) in enumerate(zip(diagrams,
                                                            zip(h_vals,eps_vals),
                                                            (4,int(4*scale_factor)),
                                                            (1,int(scale_factor))
                                                           )):
                x,y = diag.shape
                ax = plt.subplot(121+j)
                plt.imshow(diag.T,aspect="equal",origin="lower")
                plt.xticks(np.arange(offset,x,scale),axes[0][offset::scale].astype(int),fontsize="x-large")
                plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
                plt.yticks(np.arange(offset,y,scale),np.round(axes[1][offset::scale],1),fontsize="x-large")
                plt.ylabel("Energy density $\\epsilon$",fontsize="xx-large")
                plt.text(*numberpos,chr(ord('a')+j)+")",fontsize=24,color="white",horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
                if j==1:
                    # set right yticks to right side
                    plt.gca().yaxis.set_label_position("right")
                    plt.gca().yaxis.set_ticks_position("right")

            # adjust positioning
            plt.subplots_adjust(wspace=0.05)
            if savename is not None:
                plt.savefig(savename+"_{}.pdf".format(i),orientation="landscape",dpi=600,bbox_inches="tight")
            if show_plot:   
                plt.show()
            else:
                plt.close()
    return

def plot_phase_diagram(net,L_val,scale_factor,eps_train=None,N_samples=1000,h_max=30,savename=None,show_plot=False,plot_info={}):
    out = get_phase_diagram_data(net,L_val,scale_factor,eps_train,N_samples,h_max)
    plot_from_phase_diag_data(out,savename,show_plot,plot_info)
    return
