This code package contains 3 scripts: model, training and sample.

# model.py contains main architecture for variational recurrent neural network.
# training_cluster.py contains main training procedure.
# sample.py creates samples from trained latent space. (we have not used it for our experiments)

To run experiment:

## Set current directory to cadence/vrnn
## Run python training_cluster.py (for beedance), 
        python training_yahoo (for yahoo dataset), 
        python training_fishkiller (for fishkiller dataset)


####

Training parameters (number of epochs, dimensions, learning rate can be set and seen in training script.

Once trained for the selected number of epochs, results files are created and saved in a separate folders.

Currently it saves the 2 loss function values (KL divergence loss and Reconstruction loss), silhouette scores for different k and AUC scores obtained from the experiment in each epoch and plot the results after the training for the specific number of epochs.

To change the number of k for clustering, change the k values in kmeans_score function (right now k =2-5 or more is set)




