"""
Neural network model to classify MNIST digits consisting of 1 linear layer, followed by a
softmax.
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
subdirectory = "1a/"
model_filename = model_folder + subdirectory + "model_1a.ckpt"

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
tot_training_iterations = 5000
minibatch_size = 50
print_update_every = 200

# one-hot encoding converted to single number class by storing
# index of highest element.
mnist.test.klass = np.array([lbl.argmax() for lbl in mnist.test.labels])

# placeholder for data x, (784 pixels/attributes)
# placeholder for label y
x = tf.placeholder(tf.float32, [None, n_attributes])
y = tf.placeholder(tf.float32, [None, n_classes])
# placeholder for true single number class (label)
y_klass = tf.placeholder(tf.int64, [None])

# neural network model building
# define variables - initialise weights, biases as tensors full of zeros
weights = tf.Variable(tf.zeros([n_attributes, n_classes]))
biases = tf.Variable(tf.zeros([n_classes]))
# linear regression layer
y_temp = tf.add(tf.matmul(x, weights), biases)
# softmax performed
y_pred = tf.nn.softmax(y_temp)

# one-hot encoding -> predicted numerical label (class), index of largest element in row
y_pred_klass = tf.argmax(y_pred, dimension = 1)
# model is trained using cross-entropy loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
# stochastic gradient descent for minimising objective
optimiser = tf.train.GradientDescentOptimizer(optimiser_step_size).minimize(cost)
# define the accuracy - percentage of times correct prediction
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# error is 1-accuracy
error = 1 - accuracy

# save weights, biases for train.Saver
tf.add_to_collection('vars', weights)
tf.add_to_collection('vars', biases)

saver = tf.train.Saver()


"""
# NEURAL NETWORK TRAINING PROCESS

with tf.Session() as sess:
    # operation to initialise all variables
    tf.global_variables_initializer().run()
    print ("===============================")
    print("Training started........")
    # initialisation of arrays keeping track of loss/error history
    train_loss_history = np.zeros(tot_training_iterations)
    test_loss_history = np.zeros(tot_training_iterations)
    train_error_history = np.zeros(tot_training_iterations)
    test_error_history = np.zeros(tot_training_iterations)
    tot_confmat = np.zeros((n_classes,n_classes))
    # loop over each training iteration
    for iteration in range(tot_training_iterations):
        # load 'minibatch_size' examples in each training iteration
        xbatch, ybatch = mnist.train.next_batch(minibatch_size)
        # error/loss for test and train calculated for each iteration
        train_error = error.eval(feed_dict={x: xbatch, y: ybatch})
        train_error_history[iteration] = train_error
        train_loss = cost.eval(feed_dict={x: xbatch, y: ybatch})
        train_loss_history[iteration] = train_loss
        test_error = error.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
        test_error_history[iteration] = test_error
        test_loss = cost.eval(feed_dict={x: mnist.test.images,y: mnist.test.labels})
        test_loss_history[iteration] = test_loss
        # print error/loss values every 'print_update_every' iteration  to track progress
        if iteration % print_update_every == 0:
            print("iteration %d, training error %g" % (iteration, train_error))
            print("iteration %d, training loss %g" % (iteration, train_loss))
            print("iteration %d, testing error %g" % (iteration, test_error))
            print("iteration %d, testing loss %g" % (iteration, test_loss))
        # saves error/loss history to directory
        np.savez(os.getcwd() + "/traininghistory1a", train_loss_history=train_loss_history,
                 test_loss_history=test_loss_history,
                 train_error_history = train_error_history,
                 test_error_history = test_error_history)
        # run optimiser for each batch - update gradients
        optimiser.run(feed_dict={x: xbatch, y: ybatch})
        # groups test set images, one-hot encoding, true numerical labels of test set
        feed_dict_test = {x: mnist.test.images, y: mnist.test.labels, y_klass: mnist.test.klass}
        # test set true numerical classes (labels)
        true_klass = mnist.test.klass
        # test set predicted numerical classes (labels)
        pred_klass = sess.run(y_pred_klass, feed_dict=feed_dict_test)
        # sci-kit learn functionality to generate a confusion matrix
        confmat = confusion_matrix(y_true=true_klass, y_pred=pred_klass)
        tot_confmat = np.add(tot_confmat, confmat)
    # saving
    saver.save(sess, model_filename)
    print("Training finished.")
    print("===============================\n")
    # in final confusion matrix, the diagonal is set to zeros.
    # correct classifications hugely outweigh errors and are ignored in the confusion matrix plot
    for i in range(n_classes):
        tot_confmat[i,i] = 0
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
    plt.savefig('1aconfusionmatrix.png')
    ############
    # learning curve plots for cross entropy loss
    ############
    plt.figure()
    traininghistory = np.load(os.getcwd() + "/traininghistory1a.npz")
    trainingloss = traininghistory["train_loss_history"]
    testingloss = traininghistory["test_loss_history"]
    axisx = np.arange(tot_training_iterations)
    plt.plot(axisx, trainingloss, "g-", linewidth=0.8, label="training")
    plt.plot(axisx, testingloss, "r-", linewidth=0.8, label="testing")
    plt.grid()
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("cross-entropy loss")
    plt.xlim(0, tot_training_iterations)
    plt.show()
    plt.savefig('1alosslearningcurve.png')
    ###
    ###  learning curve for error
    ###
    plt.figure()
    traininghistory2 = np.load(os.getcwd() + "/traininghistory1a.npz")
    trainingerror = traininghistory2["train_error_history"]
    testingerror = traininghistory2["test_error_history"]
    axisx2 = np.arange(tot_training_iterations)
    plt.plot(axisx2, trainingerror, "c-", linewidth=0.8, label="training")
    plt.plot(axisx2, testingerror, "m-", linewidth=0.8, label="testing")
    plt.grid()
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.xlim(0, tot_training_iterations)
    plt.show()
    plt.savefig('1aerrorlearningcurve.png')

"""

### RESTORE MODEL AND OBTAIN FINAL ACCURACY

print("Restoring and testing model")
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(model_folder + subdirectory + "model_1a.ckpt.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint(model_folder + subdirectory + './'))
    all_vars = tf.get_collection('vars')
    weights = all_vars[0]
    biases = all_vars[1]
    train_error = error.eval(feed_dict={x: mnist.train.images, y: mnist.train.labels})
    test_error = error.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("The final training error is %g" % (train_error))
    print("The final test error is %g" % (test_error))










