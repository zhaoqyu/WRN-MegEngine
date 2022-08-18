#!/usr/bin/env python
# coding: utf-8

import json
from glob import glob
import numpy as np
import torchvision
import torch
import torch.nn.functional as F
from torch.utils import model_zoo


# 'https://studio.brainpp.com/api/v1/activities/3/missions/35/files/03b0e9aa-5814-4333-ba44-f848e667775c'
#params = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')
# params = torch.load(./wide-resnet-50-2-export-5ae25d50.pth)
params = model_zoo.load_url('https://studio.brainpp.com/api/v1/activities/3/missions/35/files/03b0e9aa-5814-4333-ba44-f848e667775c')


def define_model(params):
    def conv2d(input, params, base, stride=1, pad=0):
        return F.conv2d(input, params[base + '.weight'],
                        params[base + '.bias'], stride, pad)

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

    # determine network size by parameters
    blocks = [3, 4, 6, 3]

    def f(input, params):
        o = F.conv2d(input, params['conv0.weight'], params['conv0.bias'], 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = group(o, params, 'group0', 1, blocks[0])
        o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
        o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
        o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
        o = F.avg_pool2d(o_g3, 7, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f


def read_image(path):
    img = torchvision.io.read_image(path)
    img = img.float() / 255
    img = torchvision.transforms.functional.resize(img, 224)
    img = torchvision.transforms.functional.center_crop(img, 224)
    img = torchvision.transforms.functional.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return img.numpy()


data = np.stack([read_image(path) for path in glob('data/*.jpg')])
print(f'input shape {data.shape} max {data.max()} min {data.min()}')

# torch model inference
f = define_model(params)
pttensor = torch.tensor(data)
y_1 = f(pttensor, params).detach().numpy()


# meg model inference
import megengine as mge
from model import define_model

# params = mge.load("./torch2mge_model_wrn.mge")
params = mge.hub.load_serialized_obj_from_url('https://studio.brainpp.com/api/v1/activities/3/missions/35/files/1de73e5a-55c6-4813-af17-36b1fa0ef310')

f = define_model(params)
mgetensor = mge.Tensor(data)
y_2 = f(mgetensor, params).numpy()


def softmax(logits):
    logits = logits - logits.max(-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(-1, keepdims=True)

torch_output = np.argmax(softmax(y_1), axis=1)
meg_output = np.argmax(softmax(y_2), axis=1)


np.testing.assert_allclose(torch_output, meg_output, rtol=1e-3)
print('Pass')


text_labels = json.load(open('imagenet-labels.json'))

print()
print('torch')
print('megengine')
print()
for p1, p2 in zip(softmax(y_1), softmax(y_2)):
    print(text_labels[p1.argmax()], p1.max(), p1.argmax())
    print(text_labels[p2.argmax()], p2.max(), p2.argmax())
    print()
