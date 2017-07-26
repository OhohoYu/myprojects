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



def gng_1(inits, subject_no, model_regressor):
    param_no = 3
    params = np.array(["xi", "ep", "rho"])
    POI = np.array(["mu_xi", "mu_ep", "mu_rho", "sigma", "xi", "ep", "rho", "log_lik"])
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.10, 0.20, np.exp(2.0)])
        def genInitList():
            return {'mu_p': np.array([st.norm.ppf(inits_fixed[0]), st.norm.ppf(inits_fixed[1]),
                            np.log(inits_fixed[2])]),
                    'sigma': np.array([1.0, 1.0, 1.0]),
                    'xi_pr': st.norm.ppf(inits_fixed[0]).repeat(subject_no),
                    'ep_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'rho_pr': np.log(inits_fixed[2]).repeat(subject_no)}
    if model_regressor:
        POI = np.array([POI, "Qgo", "Qnogo", "Wgo", "Wnogo"])
        model_file = "gng_1_reg.stan"
    else:
        model_file = "gng_1.stan"
    return params, POI, genInitList, model_file, param_no

def gng_2(inits, subject_no, model_regressor):
    param_no = 4
    params = np.array(["xi", "ep", "b", "rho"])
    POI = np.array(["mu_xi", "mu_ep", "mu_b", "mu_rho", "sigma", "xi", "ep", "b", "rho", "log_lik"])
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.10, 0.20, 0.0, np.exp(2.0)])
        def genInitList():
            return {'mu_p': np.array([st.norm.ppf(inits_fixed[0]), st.norm.ppf(inits_fixed[1]),
                                      inits_fixed[2], np.log(inits_fixed[3])]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0]),
                    'xi_pr': st.norm.ppf(inits_fixed[0]).repeat(subject_no),
                    'ep_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'b_pr':  inits_fixed[2].repeat(subject_no),
                    'rho_pr': np.log(inits_fixed[3]).repeat(subject_no)}
    if model_regressor:
        POI = np.array([POI, "Qgo", "Qnogo", "Wgo", "Wnogo"])
        model_file = "gng_2_reg.stan"
    else:
        model_file = "gng_2.stan"
    return params, POI, genInitList, model_file, param_no

def gng_3(inits, subject_no, model_regressor):
    param_no = 5
    params = np.array(["xi", "ep", "b", "pi", "rho"])
    POI = np.array(["mu_xi", "mu_ep", "mu_b", "mu_pi", "mu_rho", "sigma", "xi", "ep", "b", "pi", "rho", "log_lik"])
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.10, 0.20, 0.10, 0.10, np.exp(2.0)])
        def genInitList():
            return {'mu_p': np.array([st.norm.ppf(inits_fixed[0]), st.norm.ppf(inits_fixed[1]),
                                      inits_fixed[2], inits_fixed[3], np.log(inits_fixed[4])]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                    'xi_pr': st.norm.ppf(inits_fixed[0]).repeat(subject_no),
                    'ep_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'b_pr':  inits_fixed[2].repeat(subject_no),
                    'pi_pr': inits_fixed[3].repeat(subject_no),
                    'rho_pr': np.log(inits_fixed[4]).repeat(subject_no)}
    if model_regressor:
        POI = np.array([POI, "Qgo", "Qnogo", "Wgo", "Wnogo", "SV"])
        model_file = "gng_3_reg.stan"
    else:
        model_file = "gng_3.stan"
    return params, POI, genInitList, model_file, param_no

def gng_4(inits, subject_no, model_regressor):
    param_no = 6
    params = np.array(["xi", "ep", "b", "pi", "mu_rhoRew", "mu_rhoPun"])
    POI = np.array(["mu_xi", "mu_ep", "mu_b", "mu_pi", "mu_rhoRew", "mu_rhoPun",
                    "sigma", "xi", "ep", "b", "pi", "rhoRew", "rhoPun", "log_lik"])
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.10, 0.20, 0.0, 0.0, np.exp(2.0), np.exp(2.0)])
        def genInitList():
            return {'mu_p': np.array([st.norm.ppf(inits_fixed[0]), st.norm.ppf(inits_fixed[1]), inits_fixed[2],
                                      inits_fixed[3], np.log(inits_fixed[4]), np.log(inits_fixed[5])]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                    'xi_pr': st.norm.ppf(inits_fixed[0]).repeat(subject_no),
                    'ep_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'b_pr': inits.fixed[2].repeat(subject_no),
                    'pi_pr': inits.fixed[3].repeat(subject_no),
                    'rhoRew_pr': np.log(inits_fixed[4]).repeat(subject_no),
                    'rhoPun_pr': np.log(inits_fixed[5]).repeat(subject_no)}
    if model_regressor:
        POI = np.array([POI, "Qgo", "Qnogo", "Wgo", "Wnogo", "SV"])
        model_file = "gng_4_reg.stan"
    else:
        model_file = "gng_4.stan"
    return params, POI, genInitList, model_file, param_no

def gng_5(inits, subject_no, model_regressor):
    param_no = 8
    params = np.array(["xi","epRew", "epPun", "b", "piRew", "piPun", "rhoRew", "rhoPun"])
    POI = np.array(["mu_xi", "mu_epRew", "mu_epPun", "mu_b", "mu_piRew", "mu_piPun", "mu_rhoRew", "mu_rhoPun",
                    "sigma", "xi", "epRew", "epPun", "b", "piRew", "piPun", "rhoRew", "rhoPun",
                    "log_lik"])
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.10, 0.20, 0.20, 0.00, 0.00, 0.00, np.exp(2.0), np.exp(2.0)])
        def genInitList():
            return {'mu_p': np.array([st.norm.ppf(inits_fixed[0]), st.norm.ppf(inits_fixed[1]),
                                      st.norm.ppf(inits_fixed[2]), inits_fixed[3], inits_fixed[4], inits_fixed[5],
                                      np.log(inits_fixed[6]), np.log(inits_fixed[7])]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                    'xi_pr': st.norm.ppf(inits_fixed[0]).repeat(subject_no),
                    'epRew_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'epPun_pr': st.norm.ppf(inits_fixed[2]).repeat(subject_no),
                    'b_pr': inits.fixed[3].repeat(subject_no),
                    'piRew_pr': inits.fixed[4].repeat(subject_no),
                    'piPun_pr': inits.fixed[5].repeat(subject_no),
                    'rhoRew_pr': np.log(inits_fixed[6]).repeat(subject_no),
                    'rhoPun_pr': np.log(inits_fixed[7]).repeat(subject_no)}
    model_file = "gng_5.stan"
    return params, POI, genInitList, model_file, param_no

def gng_6(inits, subject_no, model_regressor):
    param_no = 8
    params = np.array(["xi", "epRew", "epPun", "b", "piRew", "piPun", "rhoRew", "rhoPun"])
    POI = np.array(["mu_xi", "mu_epRew", "mu_epPun", "mu_b", "mu_piRew", "mu_piPun", "mu_rhoRew", "mu_rhoPun",
                    "sigma", "xi", "epRew", "epPun", "b", "piRew", "piPun", "rhoRew", "rhoPun",
                    "log_lik"])
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.10, 0.20, 0.20, 0.00, 0.00, 0.00, np.exp(2.0), np.exp(2.0)])
        def genInitList():
            return {'mu_p': np.array([st.norm.ppf(inits_fixed[0]), st.norm.ppf(inits_fixed[1]),
                                      st.norm.ppf(inits_fixed[2]), inits_fixed[3], inits_fixed[4], inits_fixed[5],
                                      np.log(inits_fixed[6]), np.log(inits_fixed[7])]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                    'xi_pr': st.norm.ppf(inits_fixed[0]).repeat(subject_no),
                    'epRew_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'epPun_pr': st.norm.ppf(inits_fixed[2]).repeat(subject_no),
                    'b_pr': inits.fixed[3].repeat(subject_no),
                    'piRew_pr': inits.fixed[4].repeat(subject_no),
                    'piPun_pr': inits.fixed[5].repeat(subject_no),
                    'rhoRew_pr': np.log(inits_fixed[6]).repeat(subject_no),
                    'rhoPun_pr': np.log(inits_fixed[7]).repeat(subject_no)}
    model_file = "gng_6.stan"
    return params, POI, genInitList, model_file, param_no

model_dict = { "Model_1": gng_1, "Model_2": gng_2, "Model_3": gng_3,
               "Model_4": gng_4, "Model_5": gng_5, "Model_6": gng_6 }

model_names = ['Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5', 'Model_6']

def GNG(model_name):
    return model_dict[model_name](inits, subject_no, model_regressor)