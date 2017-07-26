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
# This set-up file contains indices for the relevant columns in the data file.
# The indices can be altered for use in a different dataset. Alternatively, the dataset
# of interest can be preprocessed to match the specified set-up.
#
#
# Subject ID column -  Index of the column where subject IDs are stored. Subject IDs
# are unique identifiers for each subject within the data file.
#
subject_id_col = 1
#
# Recall that this task addresses learning of state-action contingencies trial-by-trial,
# and each trial consists of three events: a fractal cue (stored in cue_col), a target
# detection task (in this case pressing a key - keyPressed) and a probabilistic outcome.
#
# Outcome column - Index of the data file column where trial outcomes are stored. This
# script assumes outcomes take values of -1 (negative feedback), 0 (neutral feedback)
# or 1 (positive feedback) for a given trial. Such raw outcome value is 0 or 1 for win
# trials or 0 or -1 for lose trials.
#
#
rew_col = 2
#
#
# 'Key pressed" column - Index of the data file column storing the response of a
#  participant to a given trial. This script assumes responses are 1 (press/response)
#  or 0 (no press/no response).
#
los_col = 3
#
#
# 'Cue' column - Index of the data file column storing integers representing the
#  Pavlovian cues shown to the participant within a given trial. e.g. for the go/no-go
#  task, cues are made up by integers 1, 2, 3 or 4.
#
choice_col = 4