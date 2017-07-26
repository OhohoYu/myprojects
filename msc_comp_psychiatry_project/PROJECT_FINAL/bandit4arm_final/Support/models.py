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
# This file specifies parameter details, initialisation values and STAN model files
# for each RL model.
#
##########################################################################################
#
#  Model 1 ()
#
#  Model 2 ()
#
#  Model 3 ()
#
#  Model 4 ()
#
#  Model 5 ()
#
#  Model 6 ()

def bandit4a_1(inits, subject_no):
    param_no = 4
    params = np.array(["Arew", "Apun", "R", "P"])
    POI = np.array(["mu_Arew", "mu_Apun", "mu_R", "mu_P", "sigma", "Arew", "Apun", "R", "P","log_lik"])
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.10, 0.10, 1.0, 1.0])
        def genInitList():
            return {'mu_p': np.array([st.norm.ppf(inits_fixed[0]), st.norm.ppf(inits_fixed[1]),
                                      st.norm.ppf(inits_fixed[2]/30), st.norm.ppf(inits_fixed[3]/30)]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0]),
                    'Arew_pr': st.norm.ppf(inits_fixed[0]).repeat(subject_no),
                    'Apun_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'R_pr': st.norm.ppf(inits_fixed[2]/30).repeat(subject_no),
                    'P_pr': st.norm.ppf(inits_fixed[3]/30).repeat(subject_no)}
    model_file = "bandit4a_1.stan"
    return params, POI, genInitList, model_file, param_no

def bandit4a_2(inits, subject_no):
    param_no = 5
    params = np.array(["Arew", "Apun", "R", "P", "xi"])
    POI = np.array(["mu_Arew", "mu_Apun", "mu_R", "mu_P", "mu_xi", "sigma", "Arew", "Apun", "R", "P", "xi", "log_lik"])
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.10, 0.10, 1.0, 1.0, 0.10])
        def genInitList():
            return {'mu_p': np.array([st.norm.ppf(inits_fixed[0]), st.norm.ppf(inits_fixed[1]),
                                      st.norm.ppf(inits_fixed[2]/30), st.norm.ppf(inits_fixed[3]/30),
                                      st.norm.ppf(inits_fixed[4])]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                    'Arew_pr': st.norm.ppf(inits_fixed[0]).repeat(subject_no),
                    'Apun_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'R_pr': st.norm.ppf(inits_fixed[2]/30).repeat(subject_no),
                    'P_pr': st.norm.ppf(inits_fixed[3]/30).repeat(subject_no),
                    'xi_pr': st.norm.ppf(inits_fixed[4]).repeat(subject_no)}
    model_file = "bandit4a_2.stan"
    return params, POI, genInitList, model_file, param_no

model_dict = { "Model_1": bandit4a_1, "Model_2": bandit4a_2}

model_names = ['Model_1', 'Model_2']

def GNG(model_name):
    return model_dict[model_name](inits, subject_no, model_regressor)