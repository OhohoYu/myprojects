# This file implements the model described in the repository using as many  
# default settings as possible. Should the user wish to change settings, 
# various scripts and explanations of the variables are held in the
# Options_files subdirectory.
#
# Antonio Remiro, Jonathan Williams, 2016                                                      
# remiroantonio@gmail.com
# jonvw28@gmail.com    
#
################################################################################
#
# Set the location of the working directory to the directory within which the
# repository has been downloaded.
#
setwd("~/Desktop")
#
# Within this workspace, set the directory to that where the .csv files are held.
#
dir.path <- "./Data/07_05/"
#
# Set the names of the species data file (and optional distribution data file)
# NOTE: these are without '.csv' at the end
# NOTE: If you are not using a WCSP download then you will need to set manually 
# the relevant indices for your dataset in the indices settings file. 
#
spec.file.name <- "public_checklist_flat_plant_dl_20160705_poaceae"
loc.file.name <- "Poaceae_distribution"
#
# Set the name of the repository within the working directory where the output 
# will be saved. To alter the structure of the output, access the output options
# file. 
#
output.location <- "./Output"
#
# Set a unique and memorable identfier to be used in all of the output file 
# names.
#
identifier <- "exercise1"
#
# Set the start and end years
#
start.year <- 1760
end.year <- 2014
#
# Set the window size (in years)
#
interval <- 5
#
# If you would like to use rolling windows set 'rolling.windows' to TRUE and 
# specify below the number of years of the offset.  
#
rolling.windows <- FALSE
offset <- 3
#
# Specify the desired species-counting method for each time window. Select from: 
# "all", "filtered - not status", "filtered - status", 
# "filtered - status - basionym dated", 
# "filtered - includes hybrids - status - basionym dated", "all basionyms",
# "basionyms filtered - not status", "basionyms filtered - status".
#
species.method <- "filtered - status - basionym dated"
#
# Specify the desired taxonomist-counting method for each time window. Select 
# from: "all", "filtered - not status", "filtered - status", 
# "filtered - status - basionym dated", 
# "filtered - includes hybrids - status - basionym dated", "all basionyms", 
# "basionyms filtered - not status", "basionyms filtered - status".
#
taxonomist.method <- "filtered - status - basionym dated"
#
# The subsetting method below fits the model to a list of taxonomic families 
# when set to TRUE. the column where the subsetting variable is located is 
# specified in 'subset.column'. The different subsetting factors are specified
# in the vector 'subsets'. 
#
subsetting <- TRUE
subset.column <- 3
subsets <- c("Orchidaceae","Poaceae")
#
# If you would like to apply the gradient descent search method, the regression
# search cross-validation and/or the gradient descent search cross-validation, 
# set the following to TRUE. 
#
gradient.descent <- FALSE
cross.validation <- TRUE
grad.descent.cross.validation <- FALSE
#
# If you would like to apply the model to different geographic regions, set 
# 'geo.model' to TRUE. This option requires loading a second file which specifies
# the geographic locations of different species. You can also apply the gradient 
# descent search method, the regression search cross-validation and the gradient 
# descent search cross-validation at such geographic level. 
#
geo.model <- TRUE
geo.gradient.descent <- FALSE
geo.cross.validation <- FALSE
geo.grad.descent.cross.validation <- FALSE
#
# A couple of figures in Joppa et al's previous research included an additional
# data point for an incomplete time interval (2005-2008). Setting
# 'replicate.research' to TRUE, the simulation is performed including this data
# point. 'research.year' specifies a cutoff year for species publication if 
# 'replicate.research' is set to true. These two variables were used attempting 
# to replicate Joppa et al's results (published using a 2008 WCSP download). 
#
replicate.research <- FALSE
research.year <- 2008
#
#
# *MINIMUM CUMULATIVE* 
#
minimum.cumulative <- TRUE
min.c <- 50
#
# 
################################################################################
#                                                                              #
#                       DO NOT EDIT CODE BELOW THIS LINE                       #            
#                                                                              #
################################################################################
#
# Call script that selects the appropriate inputs and applies the selected models.
#
source("./kew_grasses/Support/handler.R")
#
rm(list = ls())