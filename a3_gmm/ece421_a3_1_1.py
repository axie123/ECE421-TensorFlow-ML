# -*- coding: utf-8 -*-
"""ECE421 A3 1.1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HquBRsThN0YadPubz1AG9j7R2TMVX4LW
"""

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Importing modules:

import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data from data2D.npy:

data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# Setting the # of clusters:

K = 3

is_valid = True

# Splits the dataset into training and validation set:
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Convert the data and cluster centers to tensors:

X = tf.convert_to_tensor(data, dtype=tf.float64)
MU = tf.Variable(tf.random_normal(np.array([K, dim]), dtype=X.dtype))
X = tf.expand_dims(X, 0)
MU = tf.expand_dims(MU, 1)

# Distance function for K-means:

def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    # TODO
    pair_dist = tf.transpose(tf.reduce_sum(tf.square(X - MU), axis=2))
    return pair_dist

def loss_function(X, MU):
    dist = distanceFunc(X, MU) # Get distances from data points to cluster means.
    error = tf.reduce_min(dist, axis=1) # Get smallest cluster-point distances.
    loss = tf.reduce_sum(error) # Calculates the loss of the shortest distance.
    return loss

# Assigns a cluster to the datapoint:

def cluster_assignments(X, MU):
    dist = distanceFunc(X, MU)  
    cluster = tf.argmin(dist, axis=1)
    return cluster

loss = loss_function(X, MU)
dist = distanceFunc(X, MU)
optimizer = tf.train.AdamOptimizer(learning_rate= 0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss) # Adam optimizing function.

with tf.Session() as training_loop:
  tf.initializers.global_variables().run()

  training_loss = []  # Stores the average training losses.
  validation_loss = []  # Stores the average validation losses.
  for epoch in range(150):
    new_MU, train_loss, new_dist, _ = training_loop.run([MU, loss, dist, optimizer]) # Updates hyperparameters.

    if is_valid:
      v_loss = loss_function(val_data, new_MU) # Validation loss.
      v_dist = distanceFunc(val_data, new_MU) # Validation distance.
      valid_loss, valid_dist = training_loop.run([v_loss,v_dist]) # Updates validation loss and distances.
      validation_loss.append(valid_loss/len(val_data)) 
      
    training_loss.append(train_loss/len(data))
				
    print("Epoch: ", epoch+1)
    print("Total training loss: ", train_loss)
    print("Average training loss:", train_loss / len(data))
    print("Average validation loss:", valid_loss / len(val_data))

# Training Loss Plot:

plt.plot(training_loss, label='Training Loss')
plt.xlabel('Number of Epochs')
plt.xlim(0, len(training_loss))
plt.ylabel('Average Loss')
plt.ylim(0, max(training_loss) + 1)
plt.title('Training Data Loss of K-means Clustering w/ ' + str(K) + ' Cluster Center(s)')
plt.legend()
plt.show()

# Getting the all the training distributions:
pred = np.argmin(new_dist, axis = 1) # The cluster predictions of the model.

combined_data = np.concatenate((data, pred.reshape((len(pred),1))), axis =1) # Concatenates the data with the corresponding positions.

# Getting distributions of the K clusters:
cluster_distrib_percentage = []
final_data = []
for center in range(K):
  distrib_percentage = (pred==center).sum() / data.shape[0] # Getting the percentage of each cluster.
  cluster_distrib_percentage.append(distrib_percentage)
  d = combined_data[combined_data[:,2] == center] # Getting the unique data for each of the clusters.
  final_data.append(d)

for i in range(K):
  print("Percentage of data in Cluster " + str(i + 1) + ": ", cluster_distrib_percentage[i])

# Scatter Plot of Clustering Distribution for training data:

for i in range(K):  
  plt.scatter(final_data[i][:,0], final_data[i][:,1], label = 'Cluster ' + str(i+1))
plt.plot(new_MU[:,0,0], new_MU[:,0,1], 'kx', markersize=15)
plt.xlabel('x coord data')
plt.ylabel('y coord data')
plt.title('K-means Clustering Training Data Prediction Distributions w/ '+str(K)+' Cluster Center(s)')
plt.legend()
plt.show()

# Training and Validation Loss Plot:

plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Number of Epochs')
plt.xlim(0, len(training_loss))
plt.ylabel('Average Loss')
plt.ylim(0, max(training_loss) + 1)
plt.title('Training and Validation Loss of K-means Clustering w/ '+str(K)+' Cluster Center(s)')
plt.legend()
plt.show()

# Final Validation Error of the model

with tf.Session() as get_loss:
  tf.initializers.global_variables().run()
  valid_loss = get_loss.run(v_loss)

valid_loss # Total Validation Loss

valid_loss/len(val_data) # Avg Validation Loss

# Getting the all the validation distributions:
v_pred = np.argmin(valid_dist, axis = 1) # The cluster predictions of the model.

v_combined_data = np.concatenate((val_data, v_pred.reshape((len(v_pred),1))), axis =1)

# Getting validation accuracies of the K clusters:
v_cluster_distrib_percentage = []
v_final_data = []
for center in range(K):
  v_distrib_percentage = (v_pred==center).sum() / val_data.shape[0] # Getting the percentage of each cluster.
  v_cluster_distrib_percentage.append(v_distrib_percentage)
  v_d = v_combined_data[v_combined_data[:,2] == center] # Getting the unique data for each of the clusters.
  v_final_data.append(v_d)

# Scatter Plot of Clustering Distribution for validation data:

for i in range(K):  
  plt.scatter(v_final_data[i][:,0], v_final_data[i][:,1], label = 'Cluster ' + str(i+1))
plt.plot(new_MU[:,0,0], new_MU[:,0,1], 'kx', markersize=15)
plt.xlabel('x coord data')
plt.ylabel('y coord data')
plt.title('K-means Clustering Validation Prediction Distributions w/ '+str(K)+' Cluster Center(s)')
plt.legend()
plt.show()
