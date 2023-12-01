import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt 


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


class VRNN(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim, n_layers, seq_len, bias=False):
		super(VRNN, self).__init__()

		self.x_dim = x_dim  ## 84
		self.h_dim = h_dim   ## 100, should be 50
		self.z_dim = z_dim   ## 16
		self.n_layers = n_layers  ## 1
		self.seq_len = seq_len

		#feature-extracting transformations
		##phi_x extracts features from x_t
		self.phi_x = nn.Sequential(
			nn.Linear(x_dim, h_dim), ## 84, 100 or 50
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),  ##100, 100 or 50, 50
			nn.ReLU())
		##phi_z extracts features from z_t
		self.phi_z = nn.Sequential(
			nn.Linear(z_dim, h_dim),   ## 16, 100 or 16, 50
			nn.ReLU())

		#encoder
		self.enc = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),  # 100+100, 100 or 50+50, 50
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),  # 100, 100 or 50, 50
			nn.ReLU())
		self.enc_mean = nn.Linear(h_dim, z_dim)  ## 100, 16 or 50, 16
		self.enc_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),    ## ## 100, 16 or 50, 16
			nn.Softplus())

		#prior using the hidden state
		self.prior = nn.Sequential(
			nn.Linear(h_dim, h_dim),  # 100, 100 or 50, 50
			nn.ReLU())
		self.prior_mean = nn.Linear(h_dim, z_dim) ## 100, 16 or 50, 16
		self.prior_std = nn.Sequential(       ## 100, 16 or 50, 16
			nn.Linear(h_dim, z_dim),
			nn.Softplus())

		#decoder
		self.dec = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),   # 100+100, 100 or 50+50, 50
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),     # 100, 100 or 50, 50
			nn.ReLU())
		self.dec_std = nn.Sequential(
			nn.Linear(h_dim, x_dim),     # 100, 16 or 50, 16
			nn.Softplus())
		#self.dec_mean = nn.Linear(h_dim, x_dim)
		self.dec_mean = nn.Sequential(
			nn.Linear(h_dim, x_dim),      ## 100, 84 or 50, 84
			nn.Sigmoid())  

		#recurrence
		self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)    ## 100+100, 100, 1, bias


	def forward(self, x):

		all_enc_mean, all_enc_std = [], [] 
		all_dec_mean, all_dec_std = [], [] 
		kld_loss = 0
		nll_loss = 0
		#cluster_loss=0

		h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim))
		print('h', h.size()) ## [1, 128, 100] -> layer, x-size(1), h-dim
		for t in range(self.seq_len):
			phi_x_t = self.phi_x(x[t])  ## does this 28 times, mapping between h(t-1) and x
			print('phi_x', phi_x_t.size())  ##[128, 100] --> expressing x in h_dim
			

			#encoder
			enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1)) # 128, 100
			print('enc_t', enc_t.size())  ## expressing 
			enc_mean_t = self.enc_mean(enc_t)  ## [128, 16]
			print('enc_mean_t', enc_mean_t.size())
			enc_std_t = self.enc_std(enc_t)  ##[128, 16]
			print('enc_std_t', enc_std_t.size())

			#prior
			prior_t = self.prior(h[-1])  ## [128, 100]
			print('prior_t', prior_t.size())  
			prior_mean_t = self.prior_mean(prior_t) #[128, 16]
			print('prior_mean_t', prior_mean_t.size())
			prior_std_t = self.prior_std(prior_t)  #[128, 16]
			print('prior_std_t', prior_std_t.size())

			#sampling and reparameterization
			z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
			print('z_t', z_t.size())   ## [128, 16]
			phi_z_t = self.phi_z(z_t)
			print('phi_z_t', phi_z_t.size())  ## [128, 100]

			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			print('dec_t', dec_t.size())  ## [128, 100]
			dec_mean_t = self.dec_mean(dec_t)
			print('dec_mean_t', dec_mean_t.size())  ## 128, 28
			dec_std_t = self.dec_std(dec_t)
			print('dec_std_t', dec_std_t.size())    ## 128, 28

			#recurrence
			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)  ## last h is bias
			print('h', h.size())

			#computing losses
			kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
			#nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
			nll_loss += self._L2_norm(dec_mean_t, x[t])
			#cluster_loss += self._cluster_score(enc_mean_t)



			all_enc_std.append(enc_std_t)
			all_enc_mean.append(enc_mean_t)
			all_dec_mean.append(dec_mean_t)
			all_dec_std.append(dec_std_t)

			
		return kld_loss, nll_loss,\
			(all_enc_mean, all_enc_std),\
			(all_dec_mean, all_dec_std)


	def sample(self, seq_len):

		sample = torch.zeros(seq_len, self.x_dim) ##28x84

		h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
		for t in range(seq_len):

			#prior
			prior_t = self.prior(h[-1]) #last dimension = 100
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			#sampling and reparameterization
			z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
			phi_z_t = self.phi_z(z_t)
			
			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			#dec_std_t = self.dec_std(dec_t)

			phi_x_t = self.phi_x(dec_mean_t)

			#recurrence
			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

			sample[t] = dec_mean_t.data  ## in 28 by 3 image we get from the sampling, exh pixel is the dec_mean for the pixel (reconstructed), can sample z_t or phi_z as well


	    #print('dec_mean shape', sample.shape)

		return sample


	def reset_parameters(self, stdv=1e-1):
		for weight in self.parameters():
			weight.data.normal_(0, stdv)


	def _init_weights(self, stdv):
		pass


	def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mean) #multiply eps with std of the normal and add mean, z = mean + std * eps


	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):  ## uses KL divergence formula
		"""Using std to compute KLD"""

		kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
		#print(kld_element.shape)
		#new = 0.5 * torch.sum(kld_element)
		#print(new.shape)

		return	0.5 * torch.sum(kld_element)


	#def _nll_bernoulli(self, theta, x):  ## Bernoulli for binary vectors, use other loss
		#return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))

	def _L2_norm(self, mean, x):    ##using L2 norm distance between decoder mean and x
		return torch.sum(torch.dist(x, mean, 2))



	def _nll_gauss(self, mean, std, x):
		pass