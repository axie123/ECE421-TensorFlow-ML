import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    # TODO
    pair_dist = tf.transpose(tf.reduce_sum(tf.square(X - MU), axis=2))
    return pair_dist


def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO
    dim = tf.cast(data.shape[1], "float64")
    dist = distanceFunc(X, mu)
    sigma = tf.transpose(sigma)
    gauss_coeff = tf.log(2 * np.pi * sigma)
    pdf = -(1 / 2) * dim * gauss_coeff - (dist / (2 * sigma))
    return pdf


def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    log_pi = tf.transpose(log_pi)
    prob = log_pi + log_PDF
    prob_sum = hlp.reduce_logsumexp(prob, keep_dims=True)
    posterior = prob - prob_sum
    return posterior


def loss_function(log_PDF, log_pi):
    loss = -tf.reduce_sum(hlp.reduce_logsumexp(log_PDF + log_pi, 1, keep_dims=True), axis=0)
    return loss
