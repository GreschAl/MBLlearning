##################################################################################################
# Convenience functions to create plots
# Usage is completeley optional
##################################################################################################

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True })
plt.rc('font', family='serif')

import fssa

from MBLlearning.utils.data import get_avg, coefficient_of_determination
from MBLlearning.global_config import diction

##################################################################################################

def plot_inds(net,label_idxs=[0,1,2],eps_idx=9,savename=None,L_vals=None):
    for L in net.Lvals:
        h, singlescores, targets = net.predict_energy_wise(L)
        
        # average scores over disorder realizations
        scores = np.mean(singlescores,axis=2)
        std = np.std(singlescores,axis=2)

        h, means, stds, _ = get_avg(net,L)

        plt.figure(figsize=(8,6))
        use_even = np.arange(len(h))%2==0
        use_odd  = np.bitwise_not(use_even)
        for i,idx in enumerate(label_idxs):
            plt.subplot(221+i)
            plt.errorbar(h[use_even],scores[eps_idx,use_even,i],
                         yerr=std[eps_idx,use_even,i],fmt="r.",label="Estimation")
            plt.errorbar(h[use_odd],means[eps_idx,use_odd,i],
                         yerr=stds[eps_idx,use_odd,i],fmt="b.",label="Exact diagonalization")        
            if i==2:
                plt.xticks(np.arange(2,16,2),fontsize="x-large")
                plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
            elif i==1:
                plt.xticks(np.arange(2,16,2),fontsize="x-large")
            else:
                plt.xticks(np.arange(2,16,2),[])
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
                plt.legend(bbox_to_anchor=[1.9,0.65],fontsize="large")
        
        plt.subplots_adjust(wspace=0.05,hspace=0.1)
        if savename is not None:
            plt.savefig(savename.format(L)+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
        plt.show()
    return

def plot_inds_extrapolation(net,L_extra,epsilons,label_idxs=[0,1,2],eps_idxs=[0,9,18],savename=None):
    data = (net.data).copy()
    for L in L_extra:
        try:
            # check whether data is already loaded 
            h, singlescores, targets = net.predict_energy_wise(L)
        except:
            # if not, recreate trial disorder instances to be predicted by the network
            data_test_far = {"h":np.arange(0.5,15.1,0.5)}
            h_i_vals = []
            hcorrect = []
            inds_fake = []
            for h in data_test_far["h"]:
                realization = np.random.rand(1000,L)*2*h-h
                h_i_vals.append(realization)
                hcorrect.append(np.ones(1000)*h)
                inds_fake.append(np.zeros((1000,net.N_inds*len(net.energies))))
            data_test_far["h_i"] =  np.array(h_i_vals).reshape((-1,L))
            data_test_far["inds"] =  np.array(inds_fake).reshape((-1,net.N_inds*len(net.energies)))
            data_test_far["hcorr"] = np.array(hcorrect).flatten()

            net.data[L] = { "train": None, "test": data_test_far, "estimation": None, "parameters": None }
            h, singlescores, _ = net.predict_energy_wise(L)
            
            # average scores over disorder realizations
            scores = np.mean(singlescores,axis=2)
            std = np.std(singlescores,axis=2)

            plt.figure(figsize=(8,6))
            for i,idx in enumerate(label_idxs):
                plt.subplot(221+i)
                for fmts,eps_idx in zip( ("r.","g.","b."), eps_idxs ):
                    plt.errorbar(h,scores[eps_idx,:,i],
                                 yerr=std[eps_idx,:,i],
                                 fmt=fmts,label="$\epsilon$ = {} (est.)".format(epsilons[eps_idx]))
                    if i==2:
                        plt.xticks(np.arange(2,16,2),fontsize="x-large")
                        plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
                    elif i==1:
                        plt.xticks(np.arange(2,16,2),fontsize="x-large")
                    else:
                        plt.xticks(np.arange(2,16,2),[])
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
                        plt.legend(bbox_to_anchor=[1.9,0.65],fontsize="large")
        else:
            # average scores over disorder realizations
            scores = np.mean(singlescores,axis=2)
            std = np.std(singlescores,axis=2)

            h, means, stds, _ = get_avg(net,L)

            plt.figure(figsize=(8,6))
            use_even = np.arange(len(h))%2==0
            use_odd  = np.bitwise_not(use_even)
            for i,idx in enumerate(label_idxs):
                plt.subplot(221+i)
                for fmts,eps_idx in zip( ( ("r.","c."),("g.","y."),("b.","m.") ), eps_idxs ):
                    plt.errorbar(h[use_even],scores[eps_idx,use_even,i],
                                 yerr=std[eps_idx,use_even,i],
                                 fmt=fmts[0],label="$\epsilon$ = {} (est.)".format(epsilons[eps_idx]))
                    plt.errorbar(h[use_odd],means[eps_idx,use_odd,i],
                                 yerr=stds[eps_idx,use_odd,i],
                                 fmt=fmts[1],label="$\epsilon$ = {} (ED)".format(epsilons[eps_idx]))       
                    if i==2:
                        plt.xticks(np.arange(2,16,2),fontsize="x-large")
                        plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
                    elif i==1:
                        plt.xticks(np.arange(2,16,2),fontsize="x-large")
                    else:
                        plt.xticks(np.arange(2,16,2),[])
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
                        plt.legend(bbox_to_anchor=[1.9,0.65],fontsize="large")

        plt.subplots_adjust(wspace=0.05,hspace=0.1)
        if savename is not None:
            plt.savefig(savename.format(L)+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
        plt.show()
    net.data = data
    return

def plot_r2(net,label_idxs=[0,1,2],eps_idx=9,savename=None):
    
    plt.figure(figsize=(8,6))
    for i,idx in enumerate(label_idxs):
        plt.subplot(221+i)
        for l in net.Lvals:
            h = net.data[l]["test"]["h"]
            r_squared = coefficient_of_determination(net,l,reduce_energies=True,key="test")
            r_squared = r_squared[eps_idx]
            r_normed  = 1/(2-r_squared)
            if l!=14:
                plt.plot(h,r_normed[:,i],label="L={}".format(l))
            else:
                plt.plot(h,r_normed[:,i],linestyle="dashed",label="L={}".format(l))

        plt.hlines(0.5,0,16,colors="black",linestyles="dotted")
        if i==2:
            plt.xticks(np.arange(2,16,2),fontsize="x-large")
            plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
        elif i==1:
            plt.xticks(np.arange(2,16,2),fontsize="x-large")
        else:
            plt.xticks(np.arange(2,16,2),[])
        if i%2==1:
            # set right yticks to right side
            plt.gca().yaxis.set_label_position("right")
            plt.gca().yaxis.set_ticks_position("right")
        plt.yticks(fontsize="x-large")
        plt.ylim(0,1)
        plt.xlim(0,15.5)
        plt.ylabel("$R^2_{norm.}$(" + diction[idx] +")",fontsize="x-large")
        if i==2:
            plt.legend(bbox_to_anchor=[1.8,0.7],fontsize="large")
    plt.subplots_adjust(wspace=0.05,hspace=0.2)
    if savename is not None:
        plt.savefig(savename+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
    plt.show()
    return

def plot_N_train_losses(L,N_trains,N_epochs,prefactor,filename,savename=None):
    if isinstance(L,int):
        # load data
        losses = np.empty((2,len(N_trains)),dtype=object)
        for k,key in enumerate(["train","test"]):
            for i,N in enumerate(N_trains):
                try:
                    losses[k,i] = np.loadtxt(filename.format(key,L,N))
                except:
                    losses[k,i] = None
                    print("No data for (L,key,N_train) = ({},{},{}) found on file.".format(L,key,N))
                    print("Aborted plotting. Please provide data first via training")
                    return
                    
        for k,(key,loss) in enumerate(zip(["train","test"],losses)):
            ax = plt.subplot(121+k)
            for i,N in enumerate(N_trains):
                N_e = int(N_trains[-1]*N_epochs/N)
                num_updates = np.arange(1,N_e+1)*np.ceil(N*prefactor)
                plt.semilogy(num_updates,np.mean(loss[i],axis=-1),label="N = {}".format(N))

            plt.xlim(0,num_updates[-1])
            plt.ylim(1e-3,0.3)
            plt.grid()
            if k==1:
                # set right yticks to right side
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.set_ticks_position("right")
                plt.legend(fontsize="large")
            else:
                plt.ylabel("Mean MSE",fontsize="x-large")

            plt.xticks(np.arange(4)*10000, np.arange(4)*10,fontsize="x-large")
            plt.yticks(fontsize="x-large")            
            plt.title(key,fontsize="xx-large")

        #plt.suptitle("L = {}".format(l))
        #plt.xlabel(,fontsize="large")
        plt.text(0.0,-0.15,"Number of update steps $[\\times 10^3]$",fontsize="xx-large",
                     horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
        plt.subplots_adjust(wspace=0.1)
        if savename is not None:
            plt.savefig(savename.format(L)+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
        plt.show()
            
            
    else:
        # load data
        losses = np.empty((len(L),2,len(N_trains)),dtype=object)
        L_are_available = np.zeros(len(L),dtype=bool)
        for j,l in enumerate(L):
            for k,key in enumerate(["train","test"]):
                for i,N in enumerate(N_trains):
                    try:
                        losses[j,k,i] = np.loadtxt(filename.format(key,l,N))
                        L_are_available[j] = True
                    except:
                        losses[j,k,i] = None
                        print("No data for (L,key,N_train) = ({},{},{}) found on file.".format(l,key,N))
                        print("Skipping plotting. Please provide data first via training")
        
        num_L = np.sum(L_are_available)
        
        if num_L == 0:
            print("No eligible chain lengths provided. Stopped plotting")
            return
        elif num_L > 4:
            print("Too many chain lengths provided for a single plot. Provide less than five eligible chain lengths.")
            print("Stopping plotting after the fourth chain length.")
            num_L = 4
    
        plt.figure(figsize=(6,8*num_L/2))
        for i,(l,temp) in enumerate(zip(np.array(L)[L_are_available],losses[L_are_available])):
            # plot up to the fouth available chain lengths
            if i >= 4:
                break
            for k,(key,loss) in enumerate(zip(["train","test"],temp)):
                ax = plt.subplot(100*num_L+21+k+2*i)
                for j,N in enumerate(N_trains):
                    if loss[j] is None:
                        continue # skip in case of missing data
                    N_e = int(N_trains[-1]*N_epochs/N)
                    num_updates = np.arange(1,N_e+1)*np.ceil(N*prefactor)
                    plt.semilogy(num_updates,np.mean(loss[j],axis=-1),label="N = {}".format(N))

                plt.xlim(0,num_updates[-1])
                plt.ylim(1e-3,0.3)
                plt.grid()
                if k==1:
                    # set right yticks to right side
                    plt.gca().yaxis.set_label_position("right")
                    plt.gca().yaxis.set_ticks_position("right")
                    if i==1:
                        plt.legend(fontsize="large")
                else:
                    plt.ylabel("Mean MSE",fontsize="x-large")

                if i==num_L-1:
                    plt.xticks(np.arange(4)*10000, np.arange(4)*10,fontsize="x-large")
                else:
                    plt.xticks(np.arange(4)*10000, [],fontsize="x-large")
                plt.yticks(fontsize="x-large")
                if i==0:
                    plt.title(key,fontsize="xx-large")

                plt.text(0.15,0.9,chr(ord("a")+k+2*i)+")",fontsize=22,
                         horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

        #plt.suptitle("L = {}".format(l))
        #plt.xlabel(,fontsize="large")
        plt.text(0.0,-0.18,"Number of update steps $[\\times 10^3]$",fontsize="xx-large",
                     horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
        plt.subplots_adjust(wspace=0.1,hspace=0.05)
        if savename is not None:
            plt.savefig(savename.format("multiples")+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
        plt.show()
    return

def plot_vanilla_fssa(net,label_idxs=[0,1,2],savename=None):
    colors = ["green","orange","blue"]
    epsilons = net.energies

    inds = []
    dinds = []
    inds_nn = []
    dinds_nn = []

    for l in net.Lvals:
        h, mean, std, N = get_avg(net,L=l,key="test")
        h, scores, _ = net.predict_energy_wise(l)
        inds.append(mean)
        dinds.append(std/np.sqrt(N))
        inds_nn.append(np.mean(scores,axis=2))
        dinds_nn.append(np.std(scores,axis=2)/np.sqrt(scores.shape[2]))

    inds = np.array(inds)
    dinds = np.array(dinds)
    inds_nn = np.array(inds_nn)
    dinds_nn = np.array(dinds_nn)
    
    fit_params = np.zeros((len(epsilons),len(label_idxs),6)) # three params + errors
    
    for e,epsilon in enumerate(epsilons):
        plt.figure(figsize=(8,6))
        for i,idx in enumerate(label_idxs):
            # autoscale method
            m,s = inds[:,e,:,i], dinds[:,e,:,i]
            result = fssa.autoscale(net.Lvals,h[6:18],m[:,6:18],s[:,6:18],6,1,0.1)
            m_nn,s_nn = inds_nn[:,e,:,i], dinds[:,e,:,i]
            
            if result.success:
                print("h_c = {:.2f} +/- {:.2f}".format(result.rho,result.drho))
                print("nu  = {:.2f} +/- {:.2f}".format(result.nu,result.dnu))
                print("chi = {:.2f} +/- {:.2f}".format(result.zeta,result.dzeta))
                fit_params[e,i] = np.array([result.rho,result.drho,result.nu,result.dnu,
                                                result.zeta,result.dzeta])
            else:
                print("Autoscale not successful")

            # print scaled data
            h_crit, nu_crit, chi_crit = result.x
            data = fssa.scaledata(net.Lvals,h,m,s,*result.x)
            data_nn = fssa.scaledata(net.Lvals,h,m_nn,s_nn,*result.x)

            plt.subplot(221+i)
            for j,l in enumerate(net.Lvals):
                plt.plot(data.x[j],data.y[j],
                         color=colors[j],marker="x",ls="None",label="L={}".format(l))
                plt.plot(data_nn.x[j],data_nn.y[j],
                         color=colors[j],marker=".",ls="None")#,label="L={}, RNN".format(l))
            
            plt.xticks(fontsize="x-large")
            if i>1:
                plt.xlabel("$L^{1/ \\nu }(h-h_c)$",fontsize="xx-large")
            if i%2==1:
                # set right yticks to right side
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.set_ticks_position("right")
            plt.yticks(fontsize="x-large")
            plt.ylabel(diction[idx],fontsize="x-large")

            if i==2:
                plt.legend(bbox_to_anchor=[1.8,0.7],fontsize="x-large")
        plt.subplots_adjust(wspace=0.05,hspace=0.2)

        if savename is not None:
            plt.savefig(savename+"_eps_{}.pdf".format(epsilon),
                        orientation="landscape",dpi=600,bbox_inches="tight")
        plt.show()
        
    hc = fit_params[:,:,0].T
    dhc = fit_params[:,:,1].T
    nu = fit_params[:,:,2].T
    dnu = fit_params[:,:,3].T
    colors = ["red","blue","green"]


    for j,(data,xlabel,xupper) in enumerate(zip(([hc,dhc],[nu,dnu]),
                                                ("disorder parameter $h_c$","exponent $\\nu$"),
                                                (10,4))):
        estim, destim = data
        ls = []
        labels = []

        for i,(idx,color,crit,dcrit) in enumerate(zip(label_idxs,colors,estim,destim)):
            plt.subplot(131+i)
            for e,c,dc in zip(epsilons,crit,dcrit):
                if np.isnan(dc):
                    plt.scatter(c,e,c="gray")
                else:
                    ax = plt.scatter(c,e,c=color)
                    plt.hlines(e,c-dc,c+dc,colors=color)
            ls.append(ax)
            labels.append(diction[idx])

            plt.xlim(0,xupper)
            plt.xticks(np.arange(0,xupper+1,2),fontsize="large")
            plt.grid()
            if i==0:
                plt.yticks(np.arange(1,10)/10,fontsize="large")
                plt.ylabel("Energy density $\epsilon$",fontsize="x-large")
            else:
                plt.yticks(np.arange(1,10)/10,[])
            if i==1:
                plt.xlabel("Estimated critical {}".format(xlabel),fontsize="x-large")
            #elif i==2:
                #plt.legend(ls,labels,fontsize="large",bbox_to_anchor=(0.1,0.5))
            plt.title(diction[idx])
        plt.subplots_adjust(wspace=0.3)        
        if savename is not None:
            plt.savefig(savename+"_fssa_results_{}.pdf".format(j),orientation="landscape",
                        dpi=600,bbox_inches="tight")
        plt.show()
    
    return fit_params


def plot_extrapolated_fssa(net,Lmin=12,Lmax=20,label_idxs=[0,1,2],savename=None):
    palette = plt.get_cmap('Set1')
    epsilons = net.energies
    
    data_buffer = (net.data).copy()
    
    # generate data up to L=Lmax in steps of 2
    for L in range(Lmin,Lmax+1,2):
        data_test_far = {"h":np.arange(0.5,15.1,0.5)}
        h_i_vals = []
        hcorrect = []
        inds_fake = []
        for h in data_test_far["h"]:
            realization = np.random.rand(1000,L)*2*h-h
            h_i_vals.append(realization)
            hcorrect.append(np.ones(1000)*h)
            inds_fake.append(np.zeros((1000,net.N_inds*len(net.energies))))
        data_test_far["h_i"] =  np.array(h_i_vals).reshape((-1,L))
        data_test_far["inds"] =  np.array(inds_fake).reshape((-1,net.N_inds*len(net.energies)))
        data_test_far["hcorr"] = np.array(hcorrect).flatten()

        net.data[L] = { "train": None, "test": data_test_far, "estimation": None, "parameters": None }

    print("Using L = "+("{},"*len(net.Lvals)).format(*net.Lvals)[:-1])
    
    inds_nn = []
    dinds_nn = []

    for l in range(Lmin,Lmax+1,2):
        h, scores, _ = net.predict_energy_wise(l)
        inds_nn.append(np.mean(scores,axis=2))
        dinds_nn.append(np.std(scores,axis=2)/np.sqrt(scores.shape[2]))

    inds_nn = np.array(inds_nn)
    dinds_nn = np.array(dinds_nn)
    
    fit_params_nn = np.zeros((len(epsilons),len(label_idxs),6)) # three params + errors
    
    for e,epsilon in enumerate(epsilons):
        plt.figure(figsize=(8,6))
        for i,idx in enumerate(label_idxs):
            # autoscale method
            m,s = inds_nn[:,e,:,i], dinds_nn[:,e,:,i]
            result = fssa.autoscale(range(Lmin,Lmax+1,2),h[6:18],m[:,6:18],s[:,6:18],6,1,0.1)
            
            if result.success:
                print("h_c = {:.2f} +/- {:.2f}".format(result.rho,result.drho))
                print("nu  = {:.2f} +/- {:.2f}".format(result.nu,result.dnu))
                print("chi = {:.2f} +/- {:.2f}".format(result.zeta,result.dzeta))
                fit_params_nn[e,i] = np.array([result.rho,result.drho,result.nu,result.dnu,
                                                result.zeta,result.dzeta])
            else:
                print("Autoscale not successful")

            # print scaled data
            h_crit, nu_crit, chi_crit = result.x
            data = fssa.scaledata(range(Lmin,Lmax+1,2),h,m,s,*result.x)

            plt.subplot(221+i)
            for j,l in enumerate(range(Lmin,Lmax+1,2)):
                plt.plot(data.x[j],data.y[j],
                         color=palette(j),marker="x",ls="None",label="L={}".format(l))
            
            plt.xticks(fontsize="x-large")
            if i>1:
                plt.xlabel("$L^{1/ \\nu }(h-h_c)$",fontsize="xx-large")
            if i%2==1:
                # set right yticks to right side
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.set_ticks_position("right")
            plt.yticks(fontsize="x-large")
            plt.ylabel(diction[idx],fontsize="x-large")

            if i==2:
                plt.legend(bbox_to_anchor=[1.8,0.85],fontsize="x-large")
        plt.subplots_adjust(wspace=0.05,hspace=0.2)

        if savename is not None:
            plt.savefig(savename+"_eps_{}.pdf".format(epsilon),
                        orientation="landscape",dpi=600,bbox_inches="tight")
        plt.show()
        
    hc = fit_params_nn[:,:,0].T
    dhc = fit_params_nn[:,:,1].T
    nu = fit_params_nn[:,:,2].T
    dnu = fit_params_nn[:,:,3].T
    colors = ["red","blue","green"]


    for j,(data,xlabel,xupper) in enumerate(zip(([hc,dhc],[nu,dnu]),
                                                ("disorder parameter $h_c$","exponent $\\nu$"),
                                                (10,4))):
        estim, destim = data
        ls = []
        labels = []

        for i,(idx,color,crit,dcrit) in enumerate(zip(label_idxs,colors,estim,destim)):
            plt.subplot(131+i)
            for e,c,dc in zip(epsilons,crit,dcrit):
                if np.isnan(dc):
                    plt.scatter(c,e,c="gray")
                else:
                    ax = plt.scatter(c,e,c=color)
                    plt.hlines(e,c-dc,c+dc,colors=color)
            ls.append(ax)
            labels.append(diction[idx])

            plt.xlim(0,xupper)
            plt.xticks(np.arange(0,xupper+1,2),fontsize="large")
            plt.grid()
            if i==0:
                plt.yticks(np.arange(1,10)/10,fontsize="large")
                plt.ylabel("Energy density $\epsilon$",fontsize="x-large")
            else:
                plt.yticks(np.arange(1,10)/10,[])
            if i==1:
                plt.xlabel("Estimated critical {}".format(xlabel),fontsize="x-large")
            #elif i==2:
                #plt.legend(ls,labels,fontsize="large",bbox_to_anchor=(0.1,0.5))
            plt.title(diction[idx])
        plt.subplots_adjust(wspace=0.3)        
        if savename is not None:
            plt.savefig(savename+"_fssa_results_{}.pdf".format(j),orientation="landscape",
                        dpi=600,bbox_inches="tight")
        plt.show()
        
    net.data = data_buffer
    
    return fit_params_nn

def plot_r2_comparison(transfer,net,adversary,transfer_idxs,eps_idx=9,savename=None):
    
    plt.figure(figsize=(8,6))
    for idx in transfer_idxs:
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
            if i>1:
                plt.xticks(np.arange(2,16,2),fontsize="x-large")
                plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
            else:
                plt.xticks(np.arange(2,16,2),[])
            if i%2==1:
                # set right yticks to right side
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.set_ticks_position("right")
            plt.yticks(fontsize="x-large")
            plt.ylim(0,1)
            plt.xlim(0,15.5)
            plt.ylabel("$R^2_{norm.}$",fontsize="x-large")
            plt.text(7.5,0.25,"$L={}$".format(l),fontsize="xx-large")
            if i==1:
                plt.legend(bbox_to_anchor=[0.65,-0.45],fontsize="x-large")
    plt.subplots_adjust(wspace=0.2,hspace=0.1)
    if savename is not None:
        plt.savefig(savename+".pdf",orientation="landscape",dpi=600,bbox_inches="tight")
    plt.show()
    return

def plot_L_dependent_regression(net,label_idxs=[0,1,2],eps_idx=9,savename=None):
    for i,idx in enumerate(label_idxs):
        plt.figure(figsize=(8,6))
        
        for j,L in enumerate(net.Lvals):
            h, singlescores, targets = net.predict_energy_wise(L)

            # average scores over disorder realizations
            scores = np.mean(singlescores,axis=2)
            std = np.std(singlescores,axis=2)

            h, means, stds, _ = get_avg(net,L)
            
            use_even = np.arange(len(h))%2==0
            use_odd  = np.bitwise_not(use_even)
            
            ax = plt.subplot(221+j)
            plt.errorbar(h[use_even],scores[eps_idx,use_even,i],
                         yerr=std[eps_idx,use_even,i],fmt="r.",label="Estimation")
            plt.errorbar(h[use_odd],means[eps_idx,use_odd,i],
                         yerr=stds[eps_idx,use_odd,i],fmt="b.",label="Exact diagonalization")        
            if j==2:
                plt.xticks(np.arange(2,16,2),fontsize="x-large")
                plt.xlabel("Disorder parameter $h$",fontsize="xx-large")
            elif j==1:
                plt.xticks(np.arange(2,16,2),fontsize="x-large")
            else:
                plt.xticks(np.arange(2,16,2),[])
            #plt.text(*numberpos[i],chr(ord('a')+i)+")",fontsize=22)
            plt.grid()
            plt.xlim(1,15)
            plt.text(0.8,0.8,"$L={}$".format(L),fontsize="xx-large",
                     horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
            
            #plt.ylim(ylims_dict[i])
            if j%2==1:
                # set right yticks to right side
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.set_ticks_position("right")
            plt.yticks(fontsize="x-large")
            plt.ylabel(diction[idx],fontsize="xx-large")
            if j==2:
                plt.legend(bbox_to_anchor=[1.9,0.65],fontsize="large")
        
        plt.subplots_adjust(wspace=0.05,hspace=0.1)
        if savename is not None:
            plt.savefig(savename.format(L)+"_{}.pdf".format(i),orientation="landscape",
                        dpi=600,bbox_inches="tight")
        plt.show()
    return

