"""
Neural network model to classify MNIST digits consisting of a 3 layer convolutional
model (2 convolutional layers followed by max pooling), followed by one non-linear
layer (256 units) followed by a softmax.
"""
# for compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# data directory
data_dir = "./MNIST_data/"

# for saving the model
model_folder = "savedmodels/"
# question
subdirectory = "1d/"
model_filename = model_folder + subdirectory + "model_1d.ckpt"

# load and read MNIST data, import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
# import dataset with one-hot class encoding
print("Loading the data......")
mnist = input_data.read_data_sets(data_dir, one_hot=True)
print("Data has been loaded. ")
import tensorflow as tf

# import sklearn, matplotlib for confusion matrix generation functionality
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# import numpy for train/test accuracy/loss tracking
import numpy as np

# operating system interface to save cost/accuracy history.
import os

# set a random seed for reproducibility
tf.set_random_seed(555)

# define hyper-parameter values
n_attributes = 784
n_classes = 10
optimiser_step_size = 0.5
tot_training_iterations = 8000
minibatch_size = 50
print_update_every = 50
# number of units in hidden layer 1
n_units_h1 = 256

# one-hot encoding converted to single number class by storing
# index of highest element.
mnist.test.klass = np.array([lbl.argmax() for lbl in mnist.test.labels])

# placeholder for data x, (784 pixels/attributes)
# placeholder for label y
x = tf.placeholder(tf.float32, [None, n_attributes])
y = tf.placeholder(tf.float32, [None, n_classes])
# placeholder for true single number class (label)
y_klass = tf.placeholder(tf.int64, [None])

# function which initialises (and returns) bias variables,
# it intakes the specified shape 'config' of the variables and creates
# a tensor of bias values populated with 0.1s (FOR LINEAR LAYER)
def create_biases(config):
    init = tf.constant(0.1, shape=config)
    return tf.Variable(init)

# function which initialises (and returns) weight variables.
# it intakes the specified shape 'config' of the variables and creates
# a tensor of weights from a truncated normal distribution (SD 0.1)
def create_weights(config):
    init = tf.truncated_normal(config, stddev=0.1)
    return tf.Variable(init)

# function which intakes input x and filter W and returns and computes
# their 2D convolution, which is returned - 'SAME' padding and plain (no strides)
def convolution(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

# function which intakes input x and performs max pooling on it with 2z2 sliding
# windows - no overlapping pixels.
def pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# neural network model building
# define variables - initialise weights, biases
# convolution layer 1 w/b
Wconvol1 = create_weights([3, 3, 1, 16])
bconvol1 = create_biases([16])
# before applying layer 1, reshape x to 4d tensor
reshaped1 = tf.reshape(x, [-1,28,28,1])
# apply convolution layer 1
hiddenconv1 = tf.add(convolution(reshaped1, Wconvol1), bconvol1)
# apply max pooling 1
pooling1 = pooling(hiddenconv1)
# convolution layer 2 w/b
Wconvol2 = create_weights([3, 3, 16, 16])
bconvol2 = create_biases([16])
# apply convolution layer 2
hiddenconv2 = tf.add(convolution(pooling1, Wconvol2), bconvol2)
# apply max pooling 2
pooling2 = pooling(hiddenconv2)
# FLATTEN
reshapeflat =  tf.reshape(pooling2, [-1, 7*7*16])
# hidden layer
W_h1 = tf.Variable(0.1 * tf.random_normal([n_attributes, n_units_h1]), name="W_h1")
b_h1 = tf.Variable(tf.random_normal([n_units_h1]), name="b_h1")
# add ReLu non-linearity to hidden layer 1
hidden1 = tf.nn.relu(tf.matmul(reshapeflat, W_h1) + b_h1)
# linear layer
W = tf.Variable(0.1 * tf.random_normal([n_units_h1, n_classes]), name="W")
b = tf.Variable(tf.zeros([n_classes]), name="b")
#  put together linear layer
z = tf.add(tf.matmul(hidden1, W), b)
# softmax performed
y_pred = tf.nn.softmax(z)
# term definitions
# one-hot encoding -> predicted numerical label (class), index of largest element in row
y_pred_klass = tf.argmax(y_pred, dimension=1)
# model is trained using cross-entropy loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
# run optimiser
optimiser = tf.train.GradientDescentOptimizer(optimiser_step_size).minimize(cost)
# define the accuracy - percentage of times correct prediction
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# error is 1-accuracy
error = 1 - accuracy

# save final  weights, biases for train.Saver
tf.add_to_collection('vars', W)
tf.add_to_collection('vars', b)

saver = tf.train.Saver()

"""
# NEURAL NETWORK TRAINING
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # operation to initialise all variables
    tf.global_variables_initializer().run()
    print("===============================")
    print("Training started........")
    # initialisation of arrays keeping track of loss/error history
    train_error_history = []
    test_error_history = []
    selected = []
    tot_confmat = np.zeros((n_classes, n_classes))
    # loop over each training iteration
    for iteration in range(tot_training_iterations):
        # load 'minibatch_size' examples in each training iteration
        xbatch, ybatch = mnist.train.next_batch(minibatch_size)

        if iteration % print_update_every == 0:
            # error/loss for test and train calculated for each iteration
            train_error = error.eval(feed_dict={x: xbatch, y: ybatch})
            train_error_history.append(train_error)
            test_error = error.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
            test_error_history.append(test_error)
            selected.append(iteration)
            print("iteration %d, training error %g" % (iteration, train_error))
            print("iteration %d, testing error %g" % (iteration, test_error))
            # saves error/loss history to directory
            np.savez(os.getcwd() + "/traininghistory1d",
                     train_error_history=train_error_history,
                     test_error_history=test_error_history)
            # print error/loss values every 'print_update_every' iteration  to track progress
            # groups test set images, one-hot encoding, true numerical labels of test set
            feed_dict_test = {x: mnist.test.images, y: mnist.test.labels, y_klass: mnist.test.klass}
            # test set true numerical classes (labels)
            true_klass = mnist.test.klass
            # test set predicted numerical classes (labels)
            pred_klass = sess.run(y_pred_klass, feed_dict=feed_dict_test)
            # sci-kit learn functionality to generate a confusion matrix
            confmat = confusion_matrix(y_true=true_klass, y_pred=pred_klass)
            tot_confmat = np.add(tot_confmat, confmat)
        # run optimiser for each batch - update gradients
        optimiser.run(feed_dict={x: xbatch, y: ybatch})
    print("Training finished.")
    print("===============================\n")
    # save model
    saver.save(sess, model_filename)
    # in final confusion matrix, the diagonal is set to zeros.
    # correct classifications hugely outweigh errors and are ignored in the confusion matrix plot
    for i in range(n_classes):
        tot_confmat[i, i] = 0
    ############
    # confusion matrix plots
    ############
    # text print of confusion matrix
    print(tot_confmat)
    # image print of confusion matrix
    plt.figure()
    plt.imshow(tot_confmat, interpolation='none', cmap=plt.cm.BuGn)
    # make plot nice
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, range(n_classes))
    plt.yticks(tick_marks, range(n_classes))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig('confmat1d.png')
    ############
    # learning curve plots for loss/error
    ############
    plt.figure()
    model_history = np.load(os.getcwd() + "/traininghistory1d.npz")
    train_error = model_history["train_error_history"]
    test_error = model_history["test_error_history"]
    x_axis = np.arange(0,tot_training_iterations,print_update_every)
    plt.plot(x_axis, train_error, "g-", linewidth=0.8, label="training")
    plt.plot(x_axis, test_error, "r-", linewidth=0.8, label="testing")
    plt.grid()
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.xlim(0, tot_training_iterations)
    plt.show()
    plt.savefig('1derrorlearningcurve.png')
    ###
    ###
    ###


"""


### RESTORE MODEL AND OBTAIN FINAL ACCURACY

print("Restoring and testing model")
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(model_folder + subdirectory + "model_1d.ckpt.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint(model_folder + subdirectory + './'))
    all_vars = tf.get_collection('vars')
    weights = all_vars[0]
    biases = all_vars[1]
    train_error = error.eval(feed_dict={x: mnist.train.images, y: mnist.train.labels})
    test_error = error.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("The final train error is %g" % (train_error))
    print("The final test error is %g" % (test_error))


















