import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
import os

import matplotlib.pyplot as plt 
from model import VRNN
import numpy as np
import scipy.io as sio
import sys


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


def loader():
    trn_ratio = 0.7
    window_size = 28
    dataset = sio.loadmat('../data/beedance-1.mat')
    Y = dataset['Y']
    L = dataset['L']
    T, D = Y.shape #length of data, dimension of data

    
    #print('L', L)
    #print('T', T)
    #print('D', D)
    #print('n_trn', n_trn)

    ## creates samples of shape (window_size x data dimension)
    def window(Y, step_size, window_size):
    #print('Y shape', Y.shape)
            samples = list(Y[y: y + window_size, :] for y in range(0, Y.shape[0]-window_size, step_size))

    #samples.append(sample)
            return samples
    
    data = np.asarray(window(Y[:, :], 1, window_size))

    print(np.size(data, 0))


##Train set of 70%, test set of 30%

    def train_test(data, trn_ratio):  
    #for i in range (0, data[0]):
            n_trn = int(np.ceil(trn_ratio*np.size(data,0)))
            train_set = data[0:n_trn, :, :]
            test_set = data[n_trn+1: np.size(data,0), :,:]
            return train_set, test_set

    train_set, test_set = train_test(data, trn_ratio)

    print('Train:', train_set.shape)
    print('Test:', test_set.shape)

    #train_new = train_set.reshape(train_set.shape[0], -1)
    #test_new = test_set.reshape(test_set.shape[0], -1)
    #rint('train', train_new.shape)
    #print('test:', test_new.shape)

    return train_set, test_set


##Training
def train(epoch):
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
            

            data = Variable(data.squeeze().transpose(0, 1))
            #print('Data', data.shape)
            #forward + backward + optimize

            optimizer.zero_grad()

            ##use to train model
            #kld_loss, nll_loss, cluster_loss, (enc_mean, enc_std) , (dec_mean, dec_std) = model(data) ##fetching from model script, VRNN
            
            ##use to load saved model parameters from a certain epoch
            #model = load_checkpoint('saves/vrnn_state_dict_211.pth')

            kld_loss, nll_loss, (enc_mean, enc_std) , (dec_mean, dec_std) = model(data)
            
            loss = kld_loss + nll_loss 

            loss.backward()
            
            optimizer.step()

            #grad norm clipping, only in pytorch version >= 1.10
            nn.utils.clip_grad_norm_(model.parameters(), clip) ## use it before optimizer.step()
            


            #printing
            if batch_idx % print_every == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.4f} \t NLL Loss: {:.4f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset), ##len(data) =28, batch_idx = 100, 200..., len(train_loader) = 60000
                            100. * batch_idx / len(train_loader),  ##more percentage of batches is used or loss computation
                            kld_loss.data / batch_size,
                            nll_loss.data / batch_size))


                    #print('epoch: {} \t batch_idx * len(data): {} \t len(train_loader.dataset): {} \t len(train_loader): {} \t batch_size: {}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), len(train_loader), batch_size))
                    #print ('batch_idx', batch_idx)	

                    #sample = model.sample(28)
                    #print(sample.size())
                    #plt.imshow(sample.numpy())
                    #plt.pause(1e-6)
            train_loss += loss.data


#print('train set length:', len(train_loader))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

    dec_mean = [t.detach().numpy() for t in dec_mean]
    dec_mean = torch.FloatTensor(dec_mean)
    #print('dec_mean shape_train', dec_mean.shape)


    enc_mean = [t.detach().numpy() for t in enc_mean]
    enc_mean = torch.FloatTensor(enc_mean)
    #print('enc_mean shape_train', enc_mean.shape)
    return data, dec_mean, enc_mean

			
##function to load saved model parameters		

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def test(epoch):
    """uses test data to evaluate 
    likelihood of the model"""
    
    mean_kld_loss, mean_nll_loss = 0, 0
    for i, data in enumerate(test_loader):  
            #print('Data', data.shape)                                         
            
            #data = Variable(data)
            data = Variable(data.squeeze().transpose(0, 1))


            #model = load_checkpoint('saves/vrnn_state_dict_211.pth')
            kld_loss, nll_loss, (enc_mean, enc_std), (dec_mean, dec_std) = model(data)
            mean_kld_loss += kld_loss.data
            mean_nll_loss += nll_loss.data
            #mean_cluster_loss += cluster_loss.data


    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)
    #mean_cluster_loss /= len(test_loader.dataset)


    #print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f}, Cluster Loss = {:.4f}'.format( mean_kld_loss, mean_nll_loss, mean_clutser_loss))
    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f}'.format( mean_kld_loss, mean_nll_loss))
    #print('dec_mean {}'.format(dec_mean))
    
    dec_mean = [t.detach().numpy() for t in dec_mean]
    dec_mean = torch.FloatTensor(dec_mean)
    #print('dec_mean shape', dec_mean.shape)

    enc_mean = [t.detach().numpy() for t in enc_mean]
    enc_mean = torch.FloatTensor(enc_mean)
    
    #sample = model.sample(28)
    #plt.imshow(sample.numpy())
    #plt.pause(1e-6)
    return data, dec_mean, enc_mean


def spectral_score(mean):
    mean = Variable(mean.squeeze().transpose(1, 0))
    mean = mean.reshape((28*128, z_dim))
    mean = mean.numpy()

    n_cluster = 3
    sc = SpectralClustering(n_clusters = n_cluster, assign_labels = "discretize", random_state=0)

    # predict the cluster for each data point
    cluster_means = sc.fit_predict(mean)
    

    silhouette_avg = metrics.silhouette_score(mean, cluster_means)
    print("For n_clusters =", n_cluster, "The average spectral silhouette_score is :", silhouette_avg)

    plt.scatter(mean[:, 0], mean[:, 1], c = cluster_means, s = 50, cmap = 'viridis')
    plt.title('Training data ' + str(n_cluster) + 'clusters  after' + str(epoch) + ' epochs')
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    directory = os.path.abspath('./SilScore1')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory+'/Training-' + str(n_cluster) + '-spectral-clusters-after-' + str(epoch) +  '-epochs .png')
    
    #score = score.reshape((128, 16))
    #final = torch.from_numpy(np.array(score))
    

    return silhouette_avg

def kmeans_score(mean):
    mean = Variable(mean.squeeze().transpose(1, 0))
    mean = mean.reshape((28*128, z_dim))
    mean = mean.numpy()


    n_cluster = 3
    km = KMeans(n_clusters = n_cluster, random_state=seed)
    km.fit(mean)

    # predict the cluster for each data point
    cluster_means = km.predict(mean)


    silhouette_avg = metrics.silhouette_score(mean, cluster_means)
    print("For n_clusters =", n_cluster, "The average silhouette_score is :", silhouette_avg)

    plt.scatter(mean[:, 0], mean[:, 1], c = cluster_means, s = 50, cmap = 'viridis')
    
    plt.title('Training data ' + str(n_cluster) + 'clusters  after' + str(epoch) + ' epochs')

    e=None
    if epoch>=10:
        e = str(epoch)
    else:
        e="0"+str(epoch)
    #e = (epoch>=10)?str(epoch):("0"+str(epoch))
    directory = os.path.abspath('./SilScore1')
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.savefig(directory+'/Training-' + str(n_cluster) + '-clusters-after-' + e +  '-epochs .png')
    
    #score = score.reshape((128, 16))
    #final = torch.from_numpy(np.array(score))
    

    return silhouette_avg


	

def pca_compute(enc_mean):
    enc_mean = Variable(enc_mean.squeeze().transpose(1, 0))
    enc_mean = enc_mean.reshape((28*128, z_dim))
    enc_mean = enc_mean.numpy()

    clf = PCA(0.90)
    enc_pca = clf.fit(enc_mean)
    print(clf.explained_variance_ratio_)
    print(clf.singular_values_)




	

##hyperparameter
x_dim = 3  
h_dim = 100  ##need to change the h_dim and z_dim, previously 10
z_dim = 16
n_layers =  1
n_epochs = 100
clip = 10
learning_rate = 1e-3
batch_size = 128
seed = 128
print_every = 100
save_every = 10
test_every = 500

#manual seed
torch.manual_seed(seed)
plt.ion()


train_dataset, test_dataset = loader()
train_dataset = torch.from_numpy(train_dataset).float()
test_dataset = torch.from_numpy(test_dataset).float()


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last = True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = True)

model = VRNN(x_dim, h_dim, z_dim, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_silhoette_score1 = []
train_silhoette_score2 = []
test_silhoette_score = []


for epoch in range(1, n_epochs + 1):
	
	#training + testing
	train_data, dec, enc= train(epoch)


	train_score1 = kmeans_score(enc)
	train_score2 = spectral_score(enc)
	
	train_silhoette_score1.append(train_score1)
	train_silhoette_score2.append(train_score2)


    #if epoch%test_every == 0:
	data, dec_mean, enc_mean = test(epoch)


	#saving model
	if epoch % save_every == 1:
		fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
		checkpoint = {'model': VRNN(x_dim, h_dim, z_dim, n_layers), 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
		torch.save(checkpoint, fn)
		print('Saved model to '+fn)


##Plot training Sihouette score over all epochs

##K-means
plt.figure(figsize =(15, 10))
plt.plot(train_silhoette_score1, color = 'b')
plt.xlabel('Epochs')
plt.ylabel('Train Silhouette Score')
#plt.title('Train Silhouette score over ' + str(epoch) + ' epochs', loc='center')
plt.legend(loc='best', shadow=False)

plt.savefig('Silhouette Score-1/Kmean train_silhoeutte_score after ' + str(epoch)+ '.png')

##Spectral
plt.figure(figsize =(15, 10))
plt.plot(train_silhoette_score2, color = 'b')
plt.xlabel('Epochs')
plt.ylabel('Train Silhouette Score')
#plt.title('Train Silhouette score over ' + str(epoch) + ' epochs', loc='center')
plt.legend(loc='best', shadow=False)

plt.savefig('Silhouette Score-1/Spectral train_silhoeutte_score after ' + str(epoch)+ '.png')


#manifold_PCA(enc_mean)
visualize(data, dec_mean, enc_mean)
