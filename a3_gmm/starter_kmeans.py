import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

is_valid = True

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]


# Distance function for K-means
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

def cluster_assignments(X, MU):
    dist = distanceFunc(X, MU)
    cluster = tf.argmin(dist, axis=1)
    return cluster