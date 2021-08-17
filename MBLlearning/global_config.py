##################################################################################################
# Collection of global variables and checks, such as for CUDA-usage
# Might change heavily in the next iterations
##################################################################################################


import torch
from numpy import arange, array

DTYPE = torch.float64
USE_GPU = True

diction = {0:"Adj. gap ratio $r$",1:"KL div. (norm.)",2:"$S_{Sh}/S_{max}$",3:"$S_A$ (norm.)",4:"$\Delta$",5:"Dyn. spin frac. $\mathcal{F}$",6:"Imbalance $\mathcal{I}$",7:"$S_{corr}$"}
ylims_dict = {0:(0.35,0.56),1:(0,0.6),2:(0,0.6),3:(0,0.8),4:(0,1),5:(0,1),6:(0,1),7:(-0.1,1)}
yticks_dict = {0: array([0.4,0.5]), 1: arange(0.1,0.61,0.2), 2: arange(0.1,0.61,0.2), 3: arange(0.1,0.81,0.2), 4: arange(0.1,1.1,0.4),
               5: arange(0.1,1.1,0.4), 6: arange(0.1,1.1,0.4), 7: arange(0,1.1,0.4)}
ylims_dict_extrapolate = {0:(0.35,0.56),1:(-0.1,0.8),2:(0,0.8),3:(0,0.9),4:(-0.1,1),5:(0,1),6:(-0.1,0.95),7:(-0.1,0.8)}
yticks_dict_extrapolate = {0: array([0.4,0.5]), 1: arange(0.1,0.81,0.2), 2: arange(0.1,0.81,0.2), 3: arange(0.1,0.81,0.2), 4: arange(0,0.91,0.3),
               5: arange(0.1,1.1,0.4), 6: arange(0,0.91,0.3), 7: arange(0,0.61,0.2)}

hpos = 6.75
scaling = 0.85
numberpos = {0:(hpos,scaling*(0.56-0.35)+0.35),1:(hpos,scaling*0.6),2:(hpos,scaling*0.6),3:(hpos,scaling*0.8),4:(hpos,scaling*1),5:(hpos,scaling*1),6:(hpos,scaling*1),7:(hpos,scaling*1.1-0.1)}

classes = {'0':"deloc.",'1':"MBL",'2':"unclass.",'-1':"unclass."}

def get_config():
    global USE_GPU
    global DTYPE

    if USE_GPU and torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    return DTYPE, DEVICE

def get_plot_settings():
    global diction
    global ylims_dict
    global yticks_dict
    global hpos
    global scaling
    global numberpos
    
    return (diction,ylims_dict,yticks_dict,hpos,scaling,numberpos)

def get_plot_settings_extrapolate():
    global diction
    global ylims_dict_extrapolate
    global yticks_dict_extrapolate
    global hpos
    global scaling
    global numberpos
    
    return (diction,ylims_dict_extrapolate,yticks_dict_extrapolate,hpos,scaling,numberpos)

def get_LRP_classes():
    global classes
    return classes