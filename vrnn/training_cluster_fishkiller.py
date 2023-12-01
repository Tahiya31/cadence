import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn import manifold
from sklearn.decomposition import PCA, KernelPCA
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn import svm
from sklearn import mixture
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import silhouette_samples, silhouette_score, euclidean_distances
import matplotlib.pyplot as plt 
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

from model import VRNN
import numpy as np
import scipy.io as sio
import sys
import os
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


"""Change point detction based on Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

## parameters --
   # trn_ratio = split used for train and test set
   # window_size = size of sliding window to create samples
   # step size = timestep between consecutive samples

trn_ratio = 0.7
window_size = 28
step_size = 1
## Function for sliding window by shifting by 1

def window(Y, step_size, window_size):
	samples = list(Y[y: y + window_size, :] for y in range(0, Y.shape[0]-window_size, step_size))
	return samples


## Function for splitting data into train/test set
def train_test(data, trn_ratio):  
	n_trn = int(np.ceil(trn_ratio*np.size(data,0)))
	#print (n_trn)
	train_set = data[0:n_trn, :]
	#print ('train_set.shape)
	test_set = data[n_trn+1: np.size(data,0), :]
	#print (test_set.shape)
	return train_set, test_set

## Loading dataset 
def loader(trn_ratio, window_size):
	dataset = sio.loadmat('fishkiller/fishkiller.mat')
	Y = dataset['Y']
	L = dataset['L']
	T, D = Y.shape  #length of dataset, dimension of data
	
	print('T', T)
	print('D', D)


	##creates data samples using sliding window. Each sample has a shape of (window size X data dimension)
	
	data = np.asarray(window(Y[:, :], step_size, window_size))
	w, l, d = data.shape # w--> number of samples, l--> length of each sample, d--> dimension of dataat each timestep
	print ('W', w)
	print ('L', l)
	print ('D', d)
	data = data.reshape(w, l*d)
	print(data.shape)
	
	
	#labels = np.asarray(window(L[:, :], step_size, window_size))

	train_set, test_set = train_test(data, trn_ratio)
	#print (train_set.shape)
	#print (test_set.shape)
	#train_label, test_label = train_test(labels, trn_ratio)

	print('Train:', train_set.shape)  ## shape number of training sample x window size x data dimension
	print('Test:', test_set.shape)	  ## shape number of testing sample x window size x data dimension
	#print('Train:', train_label.shape)  ## shape number of training sample x window size x data dimension
	#print('Test:', test_label.shape)    ## shape number of testing sample x window size x data dimension
	return train_set, test_set
	#, train_label, test_label

def min_max_norm(x, min, max):
    return (x - min) / (max - min)

def mini_batchify(mini_batch_size, data):

    num_m_batch = len(data) // mini_batch_size
    mini_batched_data = [] * num_m_batch
    for i in range(num_m_batch):
        mini_batched_data.append(data[i * mini_batch_size:(i + 1) * mini_batch_size])

    return torch.stack(mini_batched_data)

	
## Function to labels samples conatining changepoint as True change point
## we are not using it anymore

def window_labels(labels):
    L_true = [0]*len(labels)
    for i in range(1, len(labels)):
        for j in range (1, len(labels[0])):
            if labels[i][j]==1:
                L_true[i]=1
    return L_true



## Training process, model's forward, bachward, optimization
def train(epoch):
	train_loss = 0
	#cluster_loss=0
	latent_code = []   ##placeholder for all batches of encoded sample
	decode = []        ##placeholder for all batches of decoded sample
	data_store = []	   ##placeholder for all batches of data sample
	
	mean_kld_loss, mean_nll_loss = 0, 0

	for batch_idx, data in enumerate(train_loader):

		print("Input:", data.shape)
		data = mini_batchify(seq_len, data)
		print("Input:", data.shape)
		data = data.view(seq_len, batch_size, x_dim)
		print("Input:", data.shape)
		#data = min_max_norm(data, data.min().item(), data.max().item())
		
		#data = Variable(data.transpose(0, -1). transpose(1,-1))  ##converting from [128, 84] to [28,128, 3]
		#print("Input:", data.shape)
		
		#forward + backward + optimize

		optimizer.zero_grad()

		##fetching from model script, VRNN
		#model = load_checkpoint('saves/beedance/vrnn_state_dict_1551.pth')   ## used to load saved model, need to comment when training

		kld_loss, nll_loss, (enc_mean, enc_std) , (dec_mean, dec_std) = model(data)
	

		loss = kld_loss + nll_loss   ## beta to be used for clustering experiment, for vanilla VAE, beta = 1
		
		loss.backward()    ## To be commented when using pre loaded model for evaluation. Uncomment when training.
		
		optimizer.step()

		nn.utils.clip_grad_norm_(model.parameters(), clip)

		#grad norm clipping, only in pytorch version >= 1.10
		 ## use it before optimizer.step()
		
		mean_kld_loss += kld_loss.data
		mean_nll_loss += nll_loss.data

		train_loss += loss.data

		## Stacking all batches of encoded, decoded and input samples for visualization
		

		enc_mean = [t.detach().numpy() for t in enc_mean]
		#print("Enc:", enc_mean.size())
		enc_mean = torch.FloatTensor(enc_mean)
		print("Enc:", enc_mean.size())
		enc_mean = torch.squeeze(enc_mean, 0)
		print("Enc:", enc_mean.size())
		enc_mean = Variable(enc_mean.transpose(1, 0))
		print("Enc:", enc_mean.size())
		

		dec_mean = [t.detach().numpy() for t in dec_mean]

		dec_mean = torch.FloatTensor(dec_mean)
		
		dec_mean = Variable(dec_mean.squeeze().transpose(1, 0))
		print("Dec:", dec_mean.size())

		data = Variable(data.transpose(1, 0))
		print("Data:", data.size())

		
		latent_code.append(enc_mean)

		decode.append(dec_mean)
		
		data_store.append(data)

	latent = torch.cat(latent_code, dim = 0)
	train_enc_mean = np.array (latent)
	print("Enc:", train_enc_mean.shape)
	

	decoded = torch.cat(decode, dim = 0)
	train_dec_mean = np.array (decoded)
	print("Dec:", train_dec_mean.shape)


	input_data = torch.cat(data_store, dim = 0)
	data_input= np.array(input_data)
	print("Data:", data_input.shape)

	KLD_loss = mean_kld_loss/len(train_loader.dataset)
	REC_loss = mean_nll_loss/len(train_loader.dataset)

	#print('Length of test', len(train_loader.dataset))	
	print('Epoch: {} \t KLD Loss: {:.4f} \t NLL Loss: {:.4f}'.format(epoch, KLD_loss, REC_loss))
	
	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))

	#print('Batch:', batch_idx)
	return data, train_dec_mean, train_enc_mean, KLD_loss, REC_loss, train_loss

			
## Function for loading pre-saved model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


## Testing process, does not train, for evaluation only

def test(epoch):
	"""uses test data to evaluate 
	likelihood of the model"""
	latent_code = []   ##placeholder for all batches of encoded sample
	decode = []        ##placeholder for all batches of decoded sample
	data_store = []	   ##placeholder for all batches of data sample
	
	mean_kld_loss, mean_nll_loss = 0, 0
	
	for i, data in enumerate(test_loader):  

		print("Input:", data.shape)
		data = mini_batchify(seq_len, data)
		print("Input:", data.shape)
		data = data.view(seq_len, batch_size1, x_dim)
		print("Input:", data.shape)

		#print('Data', data.shape)                                         
		
		#data = Variable(data)
		#data = Variable(data.squeeze().transpose(0, 1))
		#print('Data', data.shape) 
		#data = (data - data.min().data) / (data.max().data - data.min().data)

		#model = load_checkpoint('saves/vrnn_state_dict_1991.pth')
		#model = load_checkpoint('saves/vrnn_state_dict_1481.pth')
		#kld_loss, nll_loss, cluster_loss, (enc_mean, enc_std), (dec_mean, dec_std) = model(data)
		#model = load_checkpoint('saves/vrnn_state_dict_91.pth')
		kld_loss, nll_loss, (enc_mean, enc_std), (dec_mean, dec_std) = model(data)

		mean_kld_loss += kld_loss.data
		mean_nll_loss += nll_loss.data
		#mean_cluster_loss += cluster_loss.data

		enc_mean = [t.detach().numpy() for t in enc_mean]
		enc_mean = torch.FloatTensor(enc_mean)
		enc_mean = Variable(enc_mean.squeeze().transpose(1, 0))
		print("Enc:", enc_mean.size())

		dec_mean = [t.detach().numpy() for t in dec_mean]
		dec_mean = torch.FloatTensor(dec_mean)
		dec_mean = Variable(dec_mean.squeeze().transpose(1, 0))
		print("Dec:", dec_mean.size())

		data = Variable(data.transpose(1, 0))
		print("Data:", data.size())
		
		#latent_code.append(enc_mean)

		#decode.append(dec_mean)
		
		#data_store.append(data)

	#latent = torch.cat(latent_code, dim = 0)
	#test_enc_mean = np.array (latent)
	

	#decoded = torch.cat(decode, dim = 0)
	#test_dec_mean = np.array (decoded)


	#input_data = torch.cat(data_store, dim = 0)
	#data_input= np.array (input_data)

	#dec_mean = [t.detach().numpy() for t in dec_mean]
	#dec_mean = torch.FloatTensor(dec_mean)
	#print('dec_mean shape', dec_mean.shape)


	#enc_mean = [t.detach().numpy() for t in enc_mean]
	#enc_mean = torch.FloatTensor(enc_mean)

	KLD_loss = mean_kld_loss/len(test_loader.dataset)
	REC_loss = mean_nll_loss/len(test_loader.dataset)


	print('Length of test', len(test_loader.dataset))
	print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f}'.format(KLD_loss, REC_loss))
	#print('dec_mean {}'.format(dec_mean))
	

	#sample = model.sample(28)
	#plt.imshow(sample.numpy())
	#plt.pause(1e-6)
	return data, dec_mean, enc_mean, KLD_loss, REC_loss


## Function to visualize t-SNE manifold
def manifold_tsne(enc_mean, epoch):
	
	enc_mean = enc_mean.reshape(enc_mean.size(0)* enc_mean.size(1), enc_mean.size(2))


	clf = manifold.TSNE(n_components=2, perplexity = 500, n_iter = 2000)
	enc_tsne = clf.fit_transform(enc_mean)

	#print(enc_tsne.kl_divergence_)

	#print(enc_tsne.n_iter_)

	xs = enc_tsne[:, 0]
	ys = enc_tsne[:, 1]
	colors = enc_tsne[:, 0]

	plt.figure(figsize =(15,10))
	plt.scatter(enc_tsne[:, 0], enc_tsne[:, 1], c =colors, cmap = "viridis")
	plt.xlabel('dim 1')
	plt.ylabel('dim 2')
	
	plt.legend(scatterpoints=1, loc='best', shadow=False)
	plt.title("t-SNE manifold of encoded data")
	directory = os.path.abspath('./figures/tsne')
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.savefig(directory+'/tsne-manifold-after-epoch-' + str(epoch)+ '.png')

def manifold_MDS(enc_mean, epoch):
	
	enc_mean = enc_mean.reshape(enc_mean.size(0)* enc_mean.size(1), enc_mean.size(2))
	#print(enc_mean.size())
	enc_mean = enc_mean.numpy()
	#print(np.dtype(enc_mean))
	#enc_mean = [t.detach().numpy() for t in enc_mean]
	#print(enc_mean.dtype)
	#enc_mean = torch.FloatTensor(enc_mean)
	#print(enc_mean.dtype)
	enc_mean = enc_mean.astype(np.float64)
	#print(enc_mean.dtype)

	#similarities = euclidean_distances(enc_mean)

	clf = manifold.MDS(n_components=2, max_iter=1000)
	enc_mds = clf.fit_transform(enc_mean)

	#print(enc_tsne.kl_divergence_)

	#print(enc_tsne.n_iter_)

	#xs = enc_mds[:, 0]
	#ys = enc_mds[:, 1]
	colors = enc_mds[:, 0]

	plt.figure(figsize =(15,10))
	plt.scatter(enc_mds[:, 0], enc_mds[:, 1], c =colors, cmap = "viridis")
	plt.xlabel('dim 1')
	plt.ylabel('dim 2')
	
	plt.legend(scatterpoints=1, loc='best', shadow=False)
	plt.title("mds manifold of encoded data")
	directory = os.path.abspath('./figures/mds')
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.savefig(directory+'/mds-manifold-after-epoch-' + str(epoch)+ '.png')


## Finding CPD candidates using cluster labels
def cpd_candidates(cluster_labels):
	print("Cluster:", cluster_labels.shape)
	
	candidates = np.where(cluster_labels[:-1] != cluster_labels[1:])[0]
	for g in candidates:
		cluster_labels[g]=1
	#print ("intermediate:", intermediate)

	#Candidate = [g+28 for g in intermediate]
	print ("Candidate:", candidates)
	return candidates, cluster_labels
def intercluster_dist(k, mean):

	sc = KMeans(n_clusters = k, random_state=0, algorithm = 'full')
	#sc = (n_clusters = k).fit(mean)
	#labels = sc.labels_

	visualizer = InterclusterDistance(sc)

	visualizer.fit(mean)        # Fit the data to the visualizer

	#visualizer.finalize()

	directory = os.path.abspath('./figures/Cluster measures')
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.savefig(directory+'/Intercluster Distance-after-' + str(n_epochs)+'-epochs .png')
	return

def clustering_measure(k, mean):
	#sc = SpectralClustering(n_clusters = k, assign_labels="discretize",random_state=0).fit(mean)
	#sc = KMeans(n_clusters = k, random_state=0, algorithm = 'full').fit(mean)
	sc = KMeans(n_clusters = k, random_state=0, algorithm = 'full').fit(mean)
	
	labels = sc.labels_
	


	silhouette_avg = silhouette_score(mean, labels)
	dists = euclidean_distances(sc.cluster_centers_)
	#print(dists)
	sample_silhouette_values = silhouette_samples(mean, labels)
	#print("Per sample silhouette coeffcients:", sample_silhouette_values)
	sample_silhouette_values = sample_silhouette_values.reshape(-1,1)
	#print(sample_silhouette_values)

	scaler = MinMaxScaler()
	scaler.fit(sample_silhouette_values)
	score = scaler.transform(sample_silhouette_values)

	return silhouette_avg, score


## Function to perform k-means clustering	
def kmeans_score(mean):
	#mean = Variable(mean.squeeze().transpose(1, 0))
	print("K_means input shape:", mean.shape)

	mean = mean.reshape((3072*10, 16))
	#print(mean.size())

	#n_clusters = [2,3,4,5,6, 7, 8, 9, 10]
	#n_clusters = [2, 3, 4]

	#for k in n_clusters:
			
	#km = KMeans(n_clusters = k, random_state=0, algorithm = 'full').fit(mean)
	#labels = km.labels_
	#candidates, changepoints = cpd_candidates(labels)
	#sil_caldidates(0.2, sample_silhouette_values)
	#sample_silhouette_values.sort()
	#print(mean.shape)

	sil_avg1, score1 = clustering_measure(2, mean)
	sil_avg2, score2 = clustering_measure(3, mean)
	sil_avg3, score3 = clustering_measure(4, mean)
	sil_avg4, score4 = clustering_measure(5, mean)

		
	print("For n_clusters =", 2, "The average silhouette_score is :", sil_avg1)
	print("For n_clusters =", 3, "The average silhouette_score is :", sil_avg2)
	print("For n_clusters =", 4, "The average silhouette_score is :", sil_avg3)
	print("For n_clusters =", 5, "The average silhouette_score is :", sil_avg4)

		
		
	#print("Per sample silhouette coeffcients:", sil_coeff)

	#print("Top 10:", np.argmax(sample_silhouette_values))
	#print("Bottom 10:", np.argmin(sample_silhouette_values))

	
	#plt.figure(figsize =(15,10))
	#plt.scatter(mean[:, 0], mean[:, 1], c = labels, cmap = 'viridis')
	#plt.xlabel('dim 1')
	#plt.ylabel('dim 2')
	#plt.legend(scatterpoints=1, loc='best', shadow=False)
	#plt.title("kmeans of encoded data")
	#directory = os.path.abspath('./figures/k-means')
	#if not os.path.exists(directory):
		#os.makedirs(directory)
	#plt.savefig(directory+'/kmeans-clustering-for-' + str(k)+'-clusters.png')

		

	return sil_avg1, score1, sil_avg2, score2, sil_avg3, score3, sil_avg4, score4



## Finding CPD candidates using silhouette coefficient

def sil_caldidates(tol, sil_coeff):
	Candidate = []
	for i, sil in enumerate(sil_coeff):
		if sil< tol:
			Candidate.append(i)
	print ("Candidate:", Candidate)
	return 



## Function to perform spectral clustering
def spectral_score(mean):
	#mean = Variable(mean.squeeze().transpose(1, 0))  #128, 28, 16
	
	mean = mean.reshape((640,window_size*z_dim))
	
	n_cluster = 2
	sc = SpectralClustering(n_clusters = n_cluster, assign_labels = "discretize", random_state=0).fit(mean)
	
	labels = sc.labels_

	candidates = cpd_candidates(labels)		

	return labels, candidates




## Function to perform agglomore clustering
def agglomore_score(mean):
	#mean = Variable(mean.squeeze().transpose(1, 0))
	mean = mean.reshape((640, window_size*z_dim))

	n_cluster = 2
			
	ag = AgglomerativeClustering(n_clusters = n_cluster, affinity = 'euclidean').fit(mean)
			
	labels = ag.labels_

	candidates = cpd_candidates(labels)

	return labels, candidates


## Function to compute Prediction Ratio
def pr_score(gt, prediction):
    if len(gt) == 0:
        return -1
    print(len(prediction))
    print(len(gt))
    return len(prediction)/len(gt)


## Function to compute Mean Absolute Error
def mae_score(gt, prediction):
    n = len(gt)
    m = len(prediction)
    minv = [float('Inf')] * n
    for j in range(n):
        for i in range(m):
            if (abs(prediction[i] - gt[j] < abs(minv[j]))):
                minv[j] = abs(prediction[i] - gt[j])
    score = sum(minv) / n
    return score

## Function to plot pairwise feature distance, not used currently
def distance_graph(enc):
	print(np.shape(enc))
	enc = enc.reshape((640, window_size*z_dim))
	print(np.shape(enc))
	#print(enc.size(1))
	#print(enc.size(2))
	print(len(enc[0]))
	dist = []
	for l in range(0, len(enc[0])-2):
		dist[l] = np.divide(np.linalg.norm((enc[0][l]-enc[0][l+1]), keepdims=True), np.sqrt((np.linalg.norm(enc[0][l]))*(np.linalg.norm(enc[0][l+1]))))  

	plt.plot(dist)
	plt.savefig('Distance.png')
	return dist

## Function visualize input and reconstructed input
## not possible any mor as dimension of dtaa is very large
def visualize(data, dec_mean, enc_mean):
	print('data shape', data.shape)
	print('dec_mean shape', dec_mean.shape)
	print('enc_mean shape', enc_mean.shape)

	data = data.reshape(data.size(0)*data.size(1), data.size(2))
	dec_mean = dec_mean.reshape(dec_mean.size(0)*dec_mean.size(1), dec_mean.size(2))
	enc_mean = enc_mean.reshape(enc_mean.size(0)*enc_mean.size(1), enc_mean.size(2))

	print('data shape', data.shape)
	print('dec_mean shape', dec_mean.shape)
	print('enc_mean shape', enc_mean.shape)


	fig1 , (ax1, ax2,ax3) = plt.subplots(3, sharex = True, sharey = True)
	#ax1 = fig1.add_subplot()
	ax1.plot(dec_mean[:,0], color='r', label='x')
	ax1.legend(loc="upper right")
	ax2.plot(dec_mean[:,1], color='b', label='y')
	ax2.legend(loc="upper right")
	ax3.plot(dec_mean[:,2], color='g', label='phase')
	ax3.legend(loc="upper right")

	plt.title("Reconstructed data")
	plt.savefig('batch_shuffled/X_hat/X_hat after training for ' + str(epoch) + ' epochs.png')


	fig2 , (ax1, ax2,ax3) = plt.subplots(3, sharex = True, sharey = True)
	#ax1 = fig1.add_subplot()
	ax1.plot(data[:,0], color='r', label='x')
	ax1.legend(loc="upper right")
	ax2.plot(data[:,1], color='b', label='y')
	ax2.legend(loc="upper right")
	ax3.plot(data[:,2], color='g', label='phase')
	ax3.legend(loc="upper right")

	plt.title("Input data")
	plt.savefig('batch_shuffled/X/X.png')



##not used anymore
def score(probability):
	prob = np.amax(probability, axis=1)
	return prob

##not used anymore
def gmm_score(mean):
	#mean = Variable(mean.squeeze().transpose(1, 0))
	mean = mean.reshape((640, window_size*z_dim))

	component = 2
			
	gmm = mixture.GaussianMixture(n_components=component, covariance_type='full',max_iter=200).fit(mean)
	prob = gmm.predict_proba(mean)
	score = gmm.score_samples(mean)
	labels = gmm.predict(mean)

	candidates = cpd_caldidates(labels)

	return prob, score, labels, candidates



def roc(labels, L):
	#print(labels)
	  ##prediction vector for change point
	#print("Check", check)

	intermediate = np.where(labels[:-1] != labels[1:])[0] ##finding where cluster labels change
	print ("intermediate:", intermediate)

	#Candidate = [g+28 for g in intermediate] ## finding corresponding new point that caused the change

	#print("Candidate:", Candidate)
	#print (len(Candidate))

	#replacements = [1]*len(Candidate)
	#print(replacements)

	#for (candid, replacement) in zip(Candidate, replacements):
		#check[candid] = replacement   ##stting the change point location in the prediction vector
	#print("Check", check)


	true_l = L[:640]
	#true_l = np.asarray(true_l)
	print(true_l)
	print(check)
	fp_list, tp_list, thresholds = metrics.roc_curve(true_l, check)
	auc = metrics.auc(fp_list, tp_list)
	print("AUC", auc)
    
	#roc_auc = auc(fp_list, tp_list)
	#plt.figure(figsize =(15, 10))
	#plt.plot(fp_list, tp_list, lw=1, alpha=0.3)
	#plt.xlabel('FPR')
	#plt.ylabel('TPR')
	#plt.title('Train Silhouette score over ' + str(epoch) + ' epochs', loc='center')
	#plt.legend(loc='best', shadow=False)

	#plt.savefig('ROC curve')

	return auc

def pair_plot (list1, list2, list3, list4):

	list1 = np.array(list1)
	list2 = np.array(list2)
	df = pd.DataFrame({'kl div loss': list1,
						'reconstrcution loss': list2,
						'auc score': list3,
						'silhouette score': list4})
	#print(df)
	sns_plot = sns.pairplot(df)
	#sns_plot.title('Pair Plot for Loss (KL divergence, Reconsruction Loss and AUC score', loc='center')
	directory = os.path.abspath('./figures/Pairplots')
	if not os.path.exists(directory):
		os.makedirs(directory)
	sns_plot.savefig(directory+'/Pairplot of loss function, silhouette score and auc score.png')

	return

def auc_score(score):
	score = 1-score
	#print(L[:640])
	#print(score)
	AUC = roc_auc_score(L[:30720], score)
	print("AUC", AUC)

	return score, AUC

def auc_score1(score):
	score = 1-score
	#print(L[:640])
	#print(score)
	AUC = roc_auc_score(L[721:1021], score)
	print("AUC", AUC)

	return score, AUC



	

##hyperparameter
x_dim = 28
h_dim = 100  ##need to change the h_dim and z_dim, previously 10
z_dim = 16
n_layers =  1
seq_len = 10
n_epochs = 1000
clip = 10 
learning_rate = 1e-3
batch_size = 128
#batch_size1 = 30
seed = 128
print_every = 100
save_every = 10
test_every = 500


window_size = 28
beta = 100


dataset = sio.loadmat('fishkiller/fishkiller.mat')
Y = dataset['Y']
L = dataset['L']
#print(L)
#gt = np.where(L[:668]==1)[0]
#print(gt)

#manual seed
torch.manual_seed(seed)
plt.ion()

#init model + optimizer + datasets
#train_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('data', train=True, download=True,
		#transform=transforms.ToTensor()),
    #batch_size=batch_size, shuffle=True)

#test_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('data', train=False, 
		#transform=transforms.ToTensor()),
    #batch_size=batch_size, shuffle=True)

train_dataset, test_dataset = loader(trn_ratio, window_size)
train_dataset = torch.from_numpy(train_dataset).float()
test_dataset = torch.from_numpy(test_dataset).float()


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size*seq_len, shuffle=False, drop_last = True)


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*seq_len, shuffle=False, drop_last = True)

model = VRNN(x_dim, h_dim, z_dim, n_layers, seq_len)
#model = VRNN(x_dim, h_dim, z_dim, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#train_silhoette_score1 = []
#train_silhoette_score2 = []
#train_silhoette_score3 = []

#MAE = []
#PR = []

#AUC = []
total = []
kld = []
rec = []
SIL_avg1=[]
SIL_avg2=[]
SIL_avg3=[]
SIL_avg4=[]
AUC1=[]
AUC2=[]
AUC3=[]
AUC4=[]

#kld2 =[]
#rec2= []
#SIL_test1=[]
#SIL_test2=[]
#SIL_test3=[]
#SIL_test4=[]
#AUC_test1=[]
#AUC_test2=[]
#AUC_test3=[]
#AUC_test4=[]


#best_auc = -1
#best_epoch =1

for epoch in range(1, n_epochs + 1):
	
	#training + testing

	
	train_data, dec, enc, KLD_loss, REC_loss, total_loss = train(epoch)
	#test_data, dec_test, enc_test, KLD_loss2, REC_loss2 = test(epoch)
	#manifold_tsne(enc, epoch)
	#manifold_MDS(enc, epoch)
	#manifold_tsne(enc, epoch)

	sil_avg1, score1, sil_avg2, score2, sil_avg3, score3, sil_avg4, score4 = kmeans_score(enc)
	pred1, auc1 = auc_score(score1)
	pred2, auc2 = auc_score(score2)
	pred3, auc3 = auc_score(score3)
	pred4, auc4 = auc_score(score4)

	#sil_test1, score_test1, sil_test2, score_test2, sil_test3, score_test3, sil_test4, score_test4 = kmeans_score(enc_test)
	#pred_test1, auc_test1 = auc_score1(score_test1)
	#pred_test2, auc_test2 = auc_score1(score_test2)
	#pred_test3, auc_test3 = auc_score1(score_test3)
	#pred_test4, auc_test4 = auc_score1(score_test4)

	#plt.figure(figsize =(15, 10))
	#n, bins, patches = plt.hist(pred4, 40, density=True, histtype='stepfilled',
                           #cumulative=True, facecolor='g', alpha=1.0, label='Empirical')
	#plt.xlabel("Changepoint Scores")
	#plt.ylabel("CDF")
	#plt.title("CDF plot of changepoint score (k=5)")
	#directory = os.path.abspath('./figures/Cluster measures')
	#if not os.path.exists(directory):
		#os.makedirs(directory)
	#plt.savefig(directory+'/Changepoint scores-after-' + str(n_epochs)+'-epochs .png')


	#gt = np.where(L[:640]==1)[0]
	#print(gt.shape)
	#print(pred3.shape)

	#print("Changepoint score from k=5:", pred4[gt])
	#print("Changepoint score from k=4:", pred3[gt])
	#print("Changepoint score from k=3:", pred2[gt])
	#print("Changepoint score from k=2:", pred1[gt])

	#print(score)

	#score = 1-score
	#print(L[:640])
	#print(score)
	#AUC = roc_auc_score(L[:640], score)
	#print("AUC", AUC)
	#prob, gmm_log_score, labels, candidates = gmm_score(enc)
	#print("Prob:", prob)
	#print("Score:", gmm_log_score)
	#pred_score = score(prob)
	#print("Score", pred_score)

	#L_train = window_labels(train_labels)
	#L_test = window_labels(test_labels)
	#d = L[:batch_size*seq_len].flatten()
	#print(d)
	#print(changepoints)
	#auc = metrics.auc(L[:batch_size*seq_len].flatten(), changepoints)
	#print("AUC:", auc)
	#test_data, dec_test, enc_test = test(epoch)
	#manifold_tsne(enc, epoch)
	#AUC.append(auc)
	kld.append(KLD_loss)
	rec.append(REC_loss)
	total.append(total_loss)


	#kld2.append(KLD_loss2)
	#rec2.append(REC_loss2)

	SIL_avg1.append(sil_avg1)
	SIL_avg2.append(sil_avg2)
	SIL_avg3.append(sil_avg3)
	SIL_avg4.append(sil_avg4)
	
	AUC1.append(auc1)
	AUC2.append(auc2)
	AUC3.append(auc3)
	AUC4.append(auc4)

	#SIL_test1.append(sil_test1)
	#SIL_test2.append(sil_test2)
	#SIL_test3.append(sil_test3)
	#SIL_test4.append(sil_test4)
	
	#AUC_test1.append(auc_test1)
	#AUC_test2.append(auc_test2)
	#AUC_test3.append(auc_test3)
	#AUC_test4.append(auc_test4)
	

	#if auc>best_auc:
		#best_auc =auc
		#best_epoch = epoch

	#saving model
	if epoch % save_every == 1:
		fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
		checkpoint = {'model': VRNN(x_dim, h_dim, z_dim, n_layers, seq_len), 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
		torch.save(checkpoint, fn)
		print('Saved model to '+fn)

#print ('best AUC at epoch {:.4f} ==> AUC:{:.4f}'.format (best_epoch, best_auc))


#pair_plot(kld, rec, AUC1, SIL_avg1)
#pair_plot(kld, rec, AUC2, SIL_avg2)
#pair_plot(kld, rec, AUC3, SIL_avg3)
#pair_plot(kld, rec, AUC4, SIL_avg4)


#plt.figure(figsize =(15, 10))
#plt.plot(AUC)
#plt.axvline(x=best_epoch, color = 'r')
#plt.xlabel('epochs')
#plt.ylabel('AUC')
#plt.title('AUC transition over' + str(n_epochs) + ' epochs', loc='center')
#directory = os.path.abspath('./figures/Performance Transition')
#if not os.path.exists(directory):
    #os.makedirs(directory)
#plt.savefig(directory+'/VRNN AUC-transition over-' + str(n_epochs)+'-epochs .png')


plt.figure(figsize =(15, 10))
plt.plot(SIL_avg1, color = 'b', label='k=2')
plt.plot(SIL_avg2, color = 'c', label='k=3')
plt.plot(SIL_avg3, color = 'k', label='k=4')
plt.plot(SIL_avg4, color = 'g', label='k=5')
plt.xlabel('epochs')
plt.ylabel('Silhouette score')
plt.title('Silhouette Score over ' + str(n_epochs) + ' epochs', loc='center')
plt.legend(loc = 'best')
directory = os.path.abspath('./figures/fishkiller')
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory+'/VRNN Silhouette Score transition over-' + str(n_epochs)+'-epochs .png')

plt.figure(figsize =(15, 10))
plt.plot(AUC1, color = 'b', label='k=2')
plt.plot(AUC2, color = 'c', label='k=3')
plt.plot(AUC3, color = 'k', label='k=4')
plt.plot(AUC4, color = 'g', label='k=5')
plt.axhline(y=0.9596, color = 'r')
plt.xlabel('epochs')
plt.ylabel('AUC score')
plt.title('AUC Score over ' + str(n_epochs) + ' epochs', loc='center')
plt.legend(loc = 'best')
directory = os.path.abspath('./figures/fishkiller')
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory+'/VRNN AUC score transition over-' + str(n_epochs)+'-epochs .png')

plt.figure(figsize =(15, 10))
plt.plot(rec, label = 'Train')
#plt.plot(rec2, label = 'Test')
plt.xlabel('epochs')
plt.ylabel('Reconstruction loss')
plt.title('Reconstruction loss transition over ' + str(n_epochs) + ' epochs', loc='center')
plt.legend(loc = 'best')
directory = os.path.abspath('./figures/fishkiller')
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory+'/VRNN Reconstruction loss transition over-' + str(n_epochs)+'-epochs.png')


plt.figure(figsize =(15, 10))
plt.plot(kld, label = 'Train')
#plt.plot(kld2, label = 'Test')
plt.xlabel('epochs')
plt.ylabel('KL divergence loss')
plt.title('KL divergence over ' + str(n_epochs) + ' epochs', loc='center')
plt.legend(loc = 'best')
directory = os.path.abspath('./figures/fishkiller')
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory+'/VRNN KLD loss transition over-' + str(n_epochs)+'-epochs.png')


plt.figure(figsize =(15, 10))
plt.plot(total, label='total_loss')
plt.xlabel('epochs')
plt.ylabel('total loss')
plt.title('total loss over' + str(n_epochs) + ' epochs', loc='center')
plt.legend(loc = 'best')
directory = os.path.abspath('./figures/fishkiller')
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory+'/VRNN total loss transition over-' + str(n_epochs)+'-epochs.png')



	#manifold_tsne(enc)
	#visualize(train_data, dec, enc)
	#manifold_tsne(enc)

	
	#labels1, candidate1 = kmeans_score(enc)
	#labels2, candidate2 = spectral_score(enc)
	#labels3, candidate3 = agglomore_score(enc)
	

	#score1 = roc(labels1, L)
	#score2 = roc(labels2, L)
	#score3 = roc(labels3, L)


	#PR1 = pr_score(gt, candidate1)
	#PR2 = pr_score(gt, candidate2)
	#PR3 = pr_score(gt, candidate3)

	#mae1 = mae_score(gt, candidate1)
	

	
	#MAE.append(mae1)
	#PR.append(PR1)


	#mae2= mae_score(gt, candidate2)
	#mae3 = mae_score(gt, candidate3)

	#print ('Kmeans ==> MAE:{:.4f}, PR: {:.4f}'.format (mae1, PR1))

	#if mae1<best_MAE:
		#best_MAE =mae1

	#print ('best ==> MAE:{:.4f}, PR: {:.4f}'.format (best_MAE, best_PR))
	#print ('Spectral ==> MAE:{:.4f}, PR: {:.4f}'.format (mae2, PR2))
	#print ('Agglomore ==> MAE:{:.4f}, PR: {:.4f}'.format (mae3, PR3))

	#dist = distance_graph(enc)
	
	#prec, rec, f, s = precision_recall_fscore_support(labels, k_means_labels, average = 'weighted')
	#print("Precision:", prec, "Recall:", rec, "F-score:", f, "Support:", s)

	#prec1, rec1, f1, s1 = precision_recall_fscore_support(labels, spectral_labels, average = 'weighted')
	#print("Precision:", prec1, "Recall:", rec1, "F-score:", f1, "Support:", s1)

	#prec2, rec2, f2, s2 = precision_recall_fscore_support(labels, agglomore_labels, average = 'weighted')
	#print("Precision:", prec2, "Recall:", rec2, "F-score:", f2, "Support:", s2)

	
	#train_silhoette_score1.append(train_score1)	
	#train_silhoette_score2.append(train_score2)
	#train_silhoette_score3.append(train_score3)
	
	#print(train_silhoette_score)
	

	

#print ('best ==> MAE:{:.4f}'.format (best_MAE))

#print(MAE)
#print(PR)
#plt.figure(figsize =(15, 10))
#plt.plot(MAE)
#plt.xlabel('epochs')
#plt.ylabel('MAE')
#plt.title('MAE transition ' + str(n_epochs) + ' epochs', loc='center')
#plt.savefig('MAE-PR/MAE over epochs.png')

#plt.figure(figsize =(15, 10))
#plt.plot(PR)
#plt.xlabel('epochs')
#plt.ylabel('PR')
#plt.title('PR transition ' + str(n_epochs) + ' epochs', loc='center')
#plt.savefig('MAE-PR/PR over  epochs.png')



#manifold_PCA(enc_mean)
#visualize(train_data, dec, enc)











