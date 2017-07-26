# This script can be utilised to replicate the go/no go task analysis made in the
# Master's Project "A Computational Psychiatry Toolbox", based at UCL in 2017.
#
# For a full explanation of the script and the methods employed, and for guidelines on
# how to use it, refer to the repository README.
#
# Antonio Remiro Azocar, 2017
# remiroantonio@gmail.com
#
########################################################################################
#
# This set-up file specifies the parameters used to draw samples from each model using
# PyStan.
#
#
# Number of chains - Positive integer specifying the number of Markov chains to be run
# to draw samples from the model.
#
chain_no = 4
#
#
# Thinning - Positive integer specifying the period for saving samples i.e. every thin_no
# samples, a posterior distribution is generated. A larger value of thin_no is useful
# when the MCMC sampling procedure presents high auto-correlation.
#
thin_no = 1
#
#
# Number of iterations - Positive integer specifying the number of iterations for each
# Markov Chain (including warmup iterations)
#
iter_no = 5000
#
#
# Number of warmup (only) iterations - Positive integer outlining the number of warm up
# (burn-in) iterations (also the number of iterations used for step-size adaptation).
# The samples drawn during warmup are not used for inference.
#
warmup_no = 2000
#
#
# Initial step size for adaptation of NUTS (no u-turn sampling) algorithm; or only step
# size if there is no adaptation).
#
stepsize = 1
#
#
# Adaptation 'delta' parameter (target acceptance rate) - Floating point value between
# 0 and 1 to control the sampling algorithm's behaviour. This value determines what the
# step size will be during sampling. The higher the acceptance rate, the lower the
# step size has to be. The lower the step size, the less likely there are to be divergent
# (numerically unstable) transitions.
#
adapt_delta = 0.95
#
#
# Maximum tree depth for the NUTS sampler (set to default value).
#
max_treedepth = 10