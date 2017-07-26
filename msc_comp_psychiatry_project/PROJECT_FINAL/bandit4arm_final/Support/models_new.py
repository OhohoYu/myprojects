def bandit4a_1():
    param_no = 4
    params = np.array(["Arew", "Apun", "R", "P"])
    POI = np.array(["mu_Arew", "mu_Apun", "mu_R", "mu_P", "sigma", "Arew", "Apun", "R", "P","log_lik"])
    model_file = "bandit4a_1.stan"
    return params, POI, model_file, param_no

def bandit4a_1_genInitList(inits, subject_no):
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
    return genInitList

def bandit4a_2():
    param_no = 5
    params = np.array(["Arew", "Apun", "R", "P", "xi"])
    POI = np.array(["mu_Arew", "mu_Apun", "mu_R", "mu_P", "mu_xi", "sigma", "Arew", "Apun", "R", "P", "xi", "log_lik"])
    model_file = "bandit4a_2.stan"
    return params, POI, model_file, param_no

def bandit4a_2_genInitList(inits, subject_no):
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.10, 0.10, 1.0, 1.0, 0.10])
        def genInitList():
            return {'mu_p': np.array([st.norm.ppf(inits_fixed[0]), st.norm.ppf(inits_fixed[1]),
                                      st.norm.ppf(inits_fixed[2]/30), st.norm.ppf(inits_fixed[3]/30)]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                    'Arew_pr': st.norm.ppf(inits_fixed[0]).repeat(subject_no),
                    'Apun_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'R_pr': st.norm.ppf(inits_fixed[2]/30).repeat(subject_no),
                    'P_pr': st.norm.ppf(inits_fixed[3]/30).repeat(subject_no),
                    'xi_pr': st.norm.ppf(inits_fixed[4]).repeat(subject_no)}
    return genInitList

model_dict = { "Model_1": bandit4a_1, "Model_2": bandit4a_2}

genInitList_dict = { "Model_1": bandit4a_1_genInitList, "Model_2": bandit4a_2_genInitList}

model_names = ['Model_1', 'Model_2']

def BANDIT4A(model_name):
    return model_dict[model_name]()

def BANDIT4A_genInitList(model_name):
    return genInitList_dict[model_name](inits, subject_no)