# This script can be utilised to replicate the go/no go task analysis made in the
# Master's Project "A Computational Psychiatry Toolbox", based at UCL in 2017.
#
# For a full explanation of the script and the methods employed, and for guidelines on
# how to use it, refer to the repository README.
#
# Antonio Remiro Azocar, 2017
# remiroantonio@gmail.com
#
##########################################################################################
#
# This file specifies some set-up options for the main functionality of the scripts.
#
#
# Number of cores - Integer designating how many CPUs the MCMC sampling is run on.
#
core_no = 1
#
#
# Initial values - String specifying the method used generate initial parameter values.
# Available options are "random" and "fixed". The "fixed" initialisation values used for
# each model can be modified in the ../Support/models.py script.
#
inits = "fixed"
#
#
# Parameter summarisation - String specifying how each model's parameters are summarised.
# Three options are available: "median", "mean" or "mode".
#=
ind_Pars = "mean"
#
#
# Model-based Regressor functionality - If True, model-based regressors e.g. Q(Go), Q(NoGo)
# are exported from the data file for posterior analysis (if available).
#
model_regressor = False
#
#
# Results directory - Path to directory where output results are saved. Output results
# include summarised model parameter values and lists of posterior samples over different
# parameters.
#
results_dir = "../Results"