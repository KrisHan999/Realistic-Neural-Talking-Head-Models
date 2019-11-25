import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from network.model import *

import os
import sys


G = Generator(224)
E = Embedder(224)
D = Discriminator(36237)
optimizerG = optim.Adam(params = list(E.parameters()) + list(G.parameters()), lr=5e-5)
optimizerD = optim.Adam(params = D.parameters(), lr=2e-4)

path_to_chkpt = 'model_weights.tar'
cpu = torch.device("cpu")

checkpoint = torch.load(path_to_chkpt, map_location=cpu)
E.load_state_dict(checkpoint['E_state_dict'])
G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
epochCurrent = checkpoint['epoch']
lossesG = checkpoint['lossesG']
lossesD = checkpoint['lossesD']
num_vid = checkpoint['num_vid']
i_batch_current = checkpoint['i_batch'] +1

data1 = pkl.load(open("data/id00017_01dfn2spqyE_00001.vid", 'rb'))
data2 = pkl.load(open("data/id00017_5MkXgwdrmJw_00002.vid", 'rb'))

fig, ax = plt.subplots(1, 2, figsize = (10, 20))
ax[0].imshow(data1[0]['frame'])
ax[1].imshow(data2[0]['frame'])

plt.show()

# The data saved in .vid is in the order of [H, W, C]
# while the model nee input of [C, W, H]
data_frame1 = np.stack([data['frame'] for data in data1], axis=0).transpose(0,3,2,1)
data_frame2 = np.stack([data['frame'] for data in data2], axis=0).transpose(0,3,2,1)
data_landmark1 = np.stack([data['landmarks'] for data in data1], axis=0).transpose(0,3,2,1)
data_landmark2 = np.stack([data['landmarks'] for data in data2], axis=0).transpose(0,3,2,1)
