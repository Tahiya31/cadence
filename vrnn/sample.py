import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import VRNN

#hyperparameters
x_dim = 3  ##28
h_dim = 20
z_dim = 16
n_layers =  1

model, state_dict, optimizer = torch.load('saves/vrnn_state_dict_41.pth')
model = VRNN(x_dim, h_dim, z_dim, n_layers)
model.load_state_dict(state_dict)

sample = model.sample(28)
print('dec_mean shape', sample.shape)

fig1 , (ax1, ax2,ax3) = plt.subplots(3, sharex = True, sharey = True)
#ax1 = fig1.add_subplot()
ax1.plot(sample[:,0], color='r', label='x')
ax1.legend(loc="upper right")
ax2.plot(sample[:,1], color='b', label='y')
ax2.legend(loc="upper right")
ax3.plot(sample[:,2], color='g', label='phase')
ax3.legend(loc="upper right")

#plt.imshow(sample.numpy())
plt.show()