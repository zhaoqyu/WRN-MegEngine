#!/usr/bin/env python
# coding: utf-8
import copy
import torch
import numpy as np
import megengine as mge
import megengine
import megengine.data as data
import megengine.data.transform as T
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
import megengine.autodiff as autodiff


import sys
from wide_resnet__mge import WideResNet
sys.path.insert(0, './')


from collections import OrderedDict
def convert2mge(torch_state, meg_state):
    new_state = OrderedDict()
    for key, value in torch_state.items():
        if len(value.size()) < 1:
            continue
        meg_value = meg_state[key]
        new_value = value.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()
        if meg_value.shape != value_np.shape:
            if meg_value.ndim == 4 and value_np.ndim == 1 and "bias" in key:
                new_value = np.reshape(value_np, meg_value.shape)
            elif meg_value.ndim == 5 and value_np.ndim == 4 and "weight" in key:
                splited = [np.expand_dims(s, axis=0) for s in np.split(value_np, meg_value.shape[0])]
                new_value = np.concatenate(splited, axis=0)
        new_state[key] = new_value
    return new_state
# torch_checkpoint = torch.load('./model/wide_resnet50_2-9ba9bcbe.pth')
# mge_model = WideResNet(Bottleneck, [3, 4, 6, 3])
# state_dict = mge_model.state_dict()
# new_state = convert2mge(torch_checkpoint,state_dict)
# mge.save(new_state, './model/wide_resnet50_2.mge')

# state_dict = mge.load("./torch2mge_lenet.mge")
# state_dict = mge.load("./code/MegEngine/mnist_079.pkl")



