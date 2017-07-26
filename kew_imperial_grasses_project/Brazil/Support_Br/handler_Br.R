# This script can be used to replicate the analysis undertaken in the summer
# project "Where are the missing grasses?" based at RBG Kew in summer 2016.
#
# For a full explanation of the script and the methods it employs, as well as
# a guide, please see the readme in this repository.
#                                                                              
# Antonio Remiro, Jonathan Williams, 2016                                                      
# remiroantonio@gmail.com    
# jonvw28@gmail.com
#
################################################################################
#
# Install any dependancies and load basic functions
#
source("./kew_grasses/Brazil/Support_Br/packages_br.R")
source("./kew_grasses/Brazil/Support_Br/functions_br.R")
#
# Load in all files giving setting for the models
#
source("./kew_grasses/Brazil/Options_Files_Br/indices_br.R")
source("./kew_grasses/Brazil/Options_Files_Br/output_options_br.R")
source("./kew_grasses/Brazil/Options_Files_Br/search_parameters_br.R")
source("./kew_grasses/Brazil/Options_Files_Br/name_formatting_br.R")
#
# Call geographical setting is requested, else set levels to NULL to signal
# not to apply geographic model
#
if(geo.model) {
        source("./kew_grasses/Options_Files_Br/geographical_model_br.R")
} else {
        levels <- NULL
        loc.ind <- NULL
        filt.ind <- NULL
        filt.mk <- NULL
        n.spec <- NULL
}
#
# Load gradient descent parameters if appropriate
#
if(gradient.descent || geo.gradient.descent) {
        source("./kew_grasses/Options_Files/gradient_descent_br.R")
}
#
# Call scripts which set parameters to ensure the correct method is applied.
#
source("./kew_grasses/Support/Data_Processing/species_method.R")
source("./kew_grasses/Support/Data_Processing/author_method.R")
#
# Call correct script to run analysis
#
if(subsetting) {
        source("./kew_grasses/Support/complete_pipeline_filter.R")
} else {
        # No filtering
        source("./kew_grasses/Support/complete_pipeline_whole_dataset.R")
}