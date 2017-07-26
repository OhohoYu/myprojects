exec(open("Support/packages.py").read()) # import?

groupings = ["A", "B", "C", "D"]

data_dir = "Data/"
data_filename = "gng_data.txt"
data = pd.read_csv(data_dir+data_filename, sep="\s+")
data = data.dropna(axis=0, how='any')
print("Rows with NAs (if any) have been removed prior to modelling the data. \n")

threat_safe_col = 7
patient_control_col = 8
threat_id = 1
safe_id = 2
patient_id = 1
control_id = 2

patients = data[data.values[:,patient_control_col-1]==patient_id]
controls = data[data.values[:,patient_control_col-1]==control_id]

threat = data[data.values[:,threat_safe_col-1]==threat_id]
safe = data[data.values[:,threat_safe_col-1]==safe_id]

patients_threat = patients[patients.values[:,threat_safe_col-1]==threat_id]
patients_safe = patients[patients.values[:,threat_safe_col-1]==safe_id]
controls_threat = controls[controls.values[:,threat_safe_col-1]==threat_id]
controls_safe = controls[controls.values[:,threat_safe_col-1]==safe_id]

exec(open("Options_Files/indices.py").read())
exec(open("Options_Files/main_opts.py").read())
exec(open("Options_Files/sampling.py").read())

"""
Orthogonalised go/no-go task
"""
time_start = time.clock()

# print task to user
# pool (multiprocessing - Philip Fowler)

subject_no = 10
exec(open("Support/models.py").read())
model_no = 6

for index in range(0, model_no):
    # remove rows with NAs
    for ind in range(0, len(groupings)):
        grouping = groupings[ind]
        if grouping == 'A':
            data_splits = [data]
        if grouping == 'B':
            data_splits = [patients, controls]
        if grouping == 'C':
            data_splits = [threat, safe]
        if grouping == 'D':
            data_splits = [patients_threat, patients_safe, controls_threat, controls_safe]
        total_looic = 0
        config_no = param_no*len(groupings)
        param_ids = list((range(param_no)))
        param_configs = [p for p in product(param_ids, repeat=2)]
        for j in range(0, len(param_configs)):
            subjects = np.unique(data_splits[inde].values[:,subject_id_col-1])

            subject_no = subjects.shape[0]
            Tsubj = (np.zeros(subject_no, dtype=np.int))
            for subj in range(0, subject_no):
                current_subj = subjects[subj]
                Tsubj[subj] = sum(data_splits[inde].values[:, subject_id_col - 1] == current_subj)
            print(Tsubj)
            max_trials = int(max(Tsubj))
            model_no = len(model_names)
            (params, POI, genInitList, model_file, param_no) = GNG(model_names[index])
            print(model_names[index], "\n")
            print("Number of Markov chains: ", chain_no, "\n")
            print("Number of cores used: ", core_no, "\n")
            print("Number of MCMC samples per chain: ", iter_no, "\n")
            print("Number of burn-in samples: ", warmup_no, "\n")
            print("Number of maximum trials per subject: ", max_trials, "\n")
            ## DATA ##
            outcome = np.zeros((subject_no, max_trials))
            pressed = np.ones((subject_no, max_trials), dtype=np.int)
            cue = np.ones((subject_no, max_trials), dtype=np.int)
            for subj in range(0, subject_no):
                current_subj = subjects[subj]
                use_Trials = Tsubj[subj]
                tmp = data_splits[inde][data_splits[inde].values[:,subject_id_col-1] == current_subj]
                outcome[subj, :use_Trials] = tmp.values[:, outcome_col-1]
                pressed[subj, :use_Trials] = tmp.values[:, keyPressed_col-1]
                cue[subj, :use_Trials] = tmp.values[:, cue_col-1]
            data_list = {'N': subject_no, 'T': max_trials, 'Tsubj': Tsubj, 'outcome': outcome,
                         'pressed': pressed, 'cue': cue}
            # Fit the STAN model
            sm = StanModel(file = model_file)
            fit = sm.sampling(data = data_list, pars = POI, warmup=warmup_no, init=genInitList, iter = iter_no,
                              chains = chain_no, thin = thin_no, control = {'adapt_delta': adapt_delta,
                                                                            'max_treedepth': max_treedepth,
                                                                            'stepsize': stepsize})
            parVals = fit.extract()
            looic, _, _ = psis.psisloo(parVals['log_lik'])  # information criterion
            total_looic += looic
        print(total_looic)












            # for ind in range(0, param_no):
            #    hist_plot = sample_hist(sample = parVals[params[ind]], title = params[ind], xlab = "Value",
            #                            ylab = "Density", xlim_min = parVals[params[ind]].min(),
            #                            xlim_max = parVals[params[ind]].max(), bin_no=25)
            #    plt.savefig(params[ind]+'.png')
            #    posterior_dist(param = params[ind], fit = fit)
            #    plt.savefig(params[ind]+'_trace_plot'+'.png')





    # fit.plot()
    # plt.savefig('foo.png', bbox_inches='tight')

    # gs = gridspec.GridSpec(param_no, 1)
    # fig = plt.figure()
    # for ind in range(param_no):
    #     param = POI[ind]
        # fig.add_subplot(gs[ind])
    #     plt.subplot(fit.plot(param))
    # fig.tight_layout()
    # plt.savefig('foo.png', bbox_inches='tight')





    # fit.plot('xi')

    # plt.tight_layout()
    # parVals = fit.extract(permuted=True)
    # with open("parVals.csv", "w") as outfile:
    #    writer = csv.writer(outfile)
    #    writer.writerow(parVals.keys())
    #    writer.writerows(zip(*parVals.values()))




    """
    xi = parVals["xi"]
    ep = parVals["ep"]
    rho = parVals["rho"]
    # extend this to other parameters
    # add parameters for each models 2, 3, 4, 5, 5b
    all_Ind_Pars = np.empty((subject_no, param_no))
    all_Ind_Pars[:] = np.NAN
    for subj in range(0,subject_no):
        if ind_Pars == "mean":
            all_Ind_Pars[subj, :] = np.array([np.mean(xi[:,subj]), np.mean(ep[:,subj]), np.mean(rho[:,subj])])
        elif ind_Pars == "median":
            all_Ind_Pars[subj, :] = np.array([np.median(xi[:,subj]), np.median(ep[:,subj]), np.median(rho[:,subj])])
        elif ind_Pars == "mode":
            all_Ind_Pars[subj, :] = np.array([st.mode(xi[:,subj]), st.mode(ep[:,subj]), st.mode(rho[:,subj])])

    subjects = np.reshape(subjects, (-1, 1))
    all_Ind_Pars = np.concatenate([all_Ind_Pars, subjects], axis=1)
    all_Ind_Pars = pd.DataFrame(all_Ind_Pars)
    all_Ind_Pars.columns = ['xi', 'ep', 'rho', 'subjID']
    model_data = {'model': model_name, 'allIndPars': all_Ind_Pars, 'parVals': parVals, 'fit': fit, 'data': data}
    # computation times
    time_elapsed = (time.clock() - time_start)
    print(time_elapsed)
    # saving functionality here
    # SEND EMAILS functionality later
    return model_data
    """






































"""
def pipeline(data, chain_no, core_no, iter_no, warmup_no, thin_no, adapt_delta, max_treedepth, stepsize,
             ind_Pars, model_name, save_results):
    time_start = time.clock()
    # remove rows with NAs
    data = data.dropna(axis=0, how='any') # check
    print("Rows with NAs (if any) have been removed prior to modelling the data. \n")
    # individual subjects
    subjects = np.unique(data.values[:,subject_id_col-1])
    subject_no = subjects.shape[0]
    # models = ["gng_1", "gng_2", "gng_3", "gng_4", "gng_5", "gng_5b"]
    # model_no = len(models)
    # for model in models:
    Tsubj = (np.zeros(subject_no, dtype=np.int))
    for subj in range(0,subject_no):
        current_subj = subjects[subj]
        Tsubj[subj] = sum(data.values[:, subject_id_col-1] == current_subj)
    max_trials = int(max(Tsubj))

    (POI, genInitList, model_file, param_no) = GNG('Model 1')
    # user info
    print("Model name = ", model_name, "\n")
    print("Number of Markov chains = ", chain_no, "\n")
    print("Number of cores used = ", core_no, "\n")
    print("Number of MCMC samples per chain = ", iter_no, "\n")
    print("Number of burn-in samples = ", warmup_no, "\n")
    print("Number of maximum trials per subject = ", max_trials, "\n")
    ## DATA ##
    outcome = np.zeros((subject_no, max_trials))
    pressed = np.ones((subject_no, max_trials), dtype=np.int)
    cue = np.ones((subject_no, max_trials), dtype=np.int)
    for subj in range(0, subject_no):
        current_subj = subjects[subj]
        use_Trials = Tsubj[subj]
        tmp = data[data.values[:,subject_id_col-1] == current_subj]
        outcome[subj, :use_Trials] = tmp.values[:, outcome_col-1]
        pressed[subj, :use_Trials] = tmp.values[:, keyPressed_col-1]
        cue[subj, :use_Trials] = tmp.values[:, cue_col-1]
    data_list = {'N': subject_no, 'T': max_trials, 'Tsubj': Tsubj, 'outcome': outcome, 'pressed': pressed, 'cue': cue}

    use multiple cores functionality (see philip fowler, do this once we have all models)

    print("***************************************")
    print("***** Loading a precompiled model *****")
    print("***************************************")
    # Fit the STAN model
    sm = StanModel(file = model_file)
    fit = sm.sampling(data = data_list, pars = POI, warmup=warmup_no, init=genInitList, iter = iter_no,
                      chains = chain_no, thin = thin_no, control = {'adapt_delta': adapt_delta,
                                                                    'max_treedepth': max_treedepth,
                                                                    'stepsize': stepsize})
    parVals = fit.extract(permuted=True)
    xi = parVals["xi"]
    ep = parVals["ep"]
    rho = parVals["rho"]
    # extend this to other parameters
    # add parameters for each models 2, 3, 4, 5, 5b
    all_Ind_Pars = np.empty((subject_no, param_no))
    all_Ind_Pars[:] = np.NAN
    for subj in range(0,subject_no):
        if ind_Pars == "mean":
            all_Ind_Pars[subj, :] = np.array([np.mean(xi[:,subj]), np.mean(ep[:,subj]), np.mean(rho[:,subj])])
        elif ind_Pars == "median":
            all_Ind_Pars[subj, :] = np.array([np.median(xi[:,subj]), np.median(ep[:,subj]), np.median(rho[:,subj])])
        elif ind_Pars == "mode":
            all_Ind_Pars[subj, :] = np.array([st.mode(xi[:,subj]), st.mode(ep[:,subj]), st.mode(rho[:,subj])])

    subjects = np.reshape(subjects, (-1, 1))
    all_Ind_Pars = np.concatenate([all_Ind_Pars, subjects], axis=1)
    all_Ind_Pars = pd.DataFrame(all_Ind_Pars)
    all_Ind_Pars.columns = ['xi', 'ep', 'rho', 'subjID']
    model_data = {'model': model_name, 'allIndPars': all_Ind_Pars, 'parVals': parVals, 'fit': fit, 'data': data}
    # computation times
    time_elapsed = (time.clock() - time_start)
    print(time_elapsed)
    # saving functionality here
    # SEND EMAILS functionality later
    return model_data

model_data = pipeline(data, chain_no, core_no, iter_no, warmup_no, thin_no, adapt_delta, max_treedepth, stepsize,
                      ind_Pars, model_name, save_results)
print(model_data)
"""


## plotting functionality
## information criteria
## hdi of mcmc

