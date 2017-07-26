# DDT
def crt_1():
    param_no = 4
    params = np.array(["alpha", "beta", "delta", "tau"])
    POI = np.array(["mu_alpha", "mu_beta", "mu_delta", "mu_tau", "sigma", "alpha", "beta", "delta", "tau", "log_lik"])
    model_file = "crt_1.stan"
    return params, POI, model_file, param_no

def crt_1_genInitList(inits, subject_no):
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.5, 0.5, 0.5, 0.15])
        def genInitList():
            return {'mu_p': np.array([np.log(inits_fixed[0]), st.norm.ppf(inits_fixed[1]),
                                      np.log(inits_fixed[2]), st.norm.ppf(inits_fixed[3])]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0]),
                    'alpha_pr': np.log(inits_fixed[0]).repeat(subject_no),
                    'beta_pr': st.norm.ppf(inits_fixed[1]).repeat(subject_no),
                    'delta_pr': np.log(inits_fixed[2]).repeat(subject_no),
                    'tau_pr': st.norm.ppf(inits_fixed[3]).repeat(subject_no)}
    return genInitList

# LBA
def crt_2():
    param_no = 4
    params = np.array(["d", "A", "v", "tau"])
    POI = np.array(["mu_d", "mu_A", "mu_v", "mu_tau", "sigma",
                    "d", "A", "v", "tau", "log_lik"])
    model_file = "crt_2.stan"
    return params, POI, model_file, param_no

def crt_2_genInitList(inits, subject_no):
    if inits == "random":
        genInitList = "random"
    elif inits == "fixed":
        inits_fixed = np.array([0.25, 0.75, 2.0, 0.2])
        def genInitList():
            return {'mu_p': np.array([inits_fixed[0], inits_fixed[1],
                                      inits_fixed[2], inits_fixed[3]]),
                    'sigma': np.array([1.0, 1.0, 1.0, 1.0]),
                    'd_pr': inits_fixed[0].repeat(subject_no),
                    'A_pr': inits_fixed[1].repeat(subject_no),
                    'v_pr': inits_fixed[2].repeat(subject_no),
                    'tau_pr': inits_fixed[3].repeat(subject_no)}
        return genInitList

# EZ diffusion
# def crt_3():
# add Model 3: EZ diffusion

model_dict = { "Model_1": crt_1, "Model_2": crt_2}

genInitList_dict = { "Model_1": crt_1_genInitList, "Model_2": crt_2_genInitList}

model_names = ['Model_1', 'Model_2']

def CRT(model_name):
    return model_dict[model_name]()

def CRT_genInitList(model_name):
    return genInitList_dict[model_name](inits, subject_no)