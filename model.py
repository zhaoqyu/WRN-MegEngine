#!/usr/bin/env python
# coding: utf-8

import megengine as mge
import megengine.functional as F
import megengine.functional.nn as nn
import megengine.module as M


def cvt_pttensor2mgetensor(pttensor):
    mgetensor = mge.Tensor(pttensor.numpy())
    return mgetensor




maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)


def broadcast(bias):
    bias = F.expand_dims(bias, axis=0)
    bias = F.expand_dims(bias, axis=2)
    bias = F.expand_dims(bias, axis=3)

    return bias

def define_model(params):
    def conv2d(input, params, base, stride=1, pad=0):
        return nn.conv2d(input, params[base + '.weight'],
                         broadcast(params[base + '.bias']), stride, pad)

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = F.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, pad=1)
            o = F.relu(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = F.relu(o)
        return o

    blocks = [3, 4, 6, 3]

    def f(input, params):
        o = nn.conv2d(input, params['conv0.weight'], broadcast(params['conv0.bias']), 2, 3)
        o = F.relu(o)
        o = maxpool(o)
        o_g0 = group(o, params, 'group0', 1, blocks[0])
        o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
        o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
        o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
        o = nn.avg_pool2d(o_g3, 7, 1, 0)
        o = F.flatten(o, 1, -1)
        o = nn.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f


# f = define_model(params)
#
# inputs_2 = cvt_pttensor2mgetensor(inputs_1)
# y_2 = f(inputs_2, params)
# print(y_1,y_2)