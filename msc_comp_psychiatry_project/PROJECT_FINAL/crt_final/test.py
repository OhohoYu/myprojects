exec(open("Support/packages.py").read()) # import?

threat_safe_col = 8
patient_control_col = 7
threat_id = 1
safe_id = 2
patient_id = 1
control_id = 2

data_dir = "Data/"
data_filename = "crt_data_1.txt"
data = pd.read_csv(data_dir+data_filename, sep="\s+")
data = data.dropna(axis=0, how='any')
print("Rows with NAs (if any) have been removed prior to modelling the data. \n")

# initially, we have data of 77 participants (no threat/safe distinction, only patient/control distinction)

patients = data[data.values[:,patient_control_col-1]==patient_id]
controls = data[data.values[:,patient_control_col-1]==control_id]

exec(open("Options_Files/indices.py").read())
exec(open("Options_Files/main_opts.py").read())
exec(open("Options_Files/sampling.py").read())

"""
Orthogonalised go/no-go task
"""
time_start = time.clock()

# print task to user
# pool (multiprocessing - Philip Fowler)

exec(open("Support/models_new.py").read())
model_no = len(model_names)

groupings = np.array([0, 1]);

for index in range(0, model_no): #model_no
    (params, POI, model_file, param_no) = CRT(model_names[index])
    group_ids = list((range(len(groupings))))
    param_configs = [p for p in product(group_ids, repeat=param_no)]
    dd_configs = []
    for ind in range(0, len(param_configs)):
        param_config = param_configs[ind]
        print(param_config)
        param_config_dd = list(OrderedSet(param_config))
        run_no = len(param_config_dd)
        data_splits = [None]*run_no
        for run in range(0, run_no):
            grouping = param_config_dd[run]
            if grouping == 0:
                data_splits = [data]
            elif grouping == 1:
                data_splits = [patients, controls]
            for k in range(0, len(data_splits)):
                subjects = np.unique(data_splits[k].values[:, subject_id_col - 1])
                subject_no = subjects.shape[0]
                Tsubj = (np.zeros(subject_no, dtype=np.int))
                for subj in range(0, subject_no):
                    current_subj = subjects[subj]
                    Tsubj[subj] = sum(data_splits[k].values[:, subject_id_col - 1] == current_subj)
                max_trials = int(max(Tsubj))
                print(model_names[index], "\n")
                print("Number of Markov chains: ", chain_no, "\n")
                print("Number of cores used: ", core_no, "\n")
                print("Number of MCMC samples per chain: ", iter_no, "\n")
                print("Number of burn-in samples: ", warmup_no, "\n")
                print("Number of maximum trials per subject: ", max_trials, "\n")
                outcome = np.zeros((subject_no, max_trials))
                pressed = np.ones((subject_no, max_trials), dtype=np.int)
                cue = np.ones((subject_no, max_trials), dtype=np.int)
                for subj in range(0, subject_no):
                    current_subj = subjects[subj]
                    use_Trials = Tsubj[subj]
                    tmp = data_splits[k][data_splits[k].values[:, subject_id_col - 1] == current_subj]
                    outcome[subj, :use_Trials] = tmp.values[:, outcome_col - 1]
                    pressed[subj, :use_Trials] = tmp.values[:, keyPressed_col - 1]
                    cue[subj, :use_Trials] = tmp.values[:, cue_col - 1]
                data_list = {'N': subject_no, 'T': max_trials, 'Tsubj': Tsubj, 'outcome': outcome,
                             'pressed': pressed, 'cue': cue, 'param_config': param_config, 'grouping': grouping}
                genInitList = GNG_genInitList(model_names[index])
                # Fit the STAN model
                sm = StanModel(file=model_file)
                fit = sm.sampling(data=data_list, pars=POI, warmup=warmup_no, init=genInitList, iter=iter_no,
                                  chains=chain_no, thin=thin_no, control={'adapt_delta': adapt_delta,
                                                                          'max_treedepth': max_treedepth,
                                                                          'stepsize': stepsize})



















































# print task to user
# pool (multiprocessing - Philip Fowler)

exec(open("Support/models_new.py").read())
model_no = len(model_names)

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

exec(open("Support/models_new.py").read())
model_no = len(model_names)

groupings = np.array([0, 1, 2, 3]);

for index in range(0, 1): #model_no
    (params, POI, model_file, param_no) = GNG(model_names[index])
    group_ids = list((range(len(groupings))))
    param_configs = [p for p in product(group_ids, repeat=param_no)]
    dd_configs = []
    for ind in range(0, len(param_configs)):
        param_config = param_configs[ind]
        print(param_config)
        param_config_dd = list(OrderedSet(param_config))
        run_no = len(param_config_dd)
        # data_splits = list((range(run_no)))
        data_splits = [None]*run_no
        for run in range(0, run_no):
            grouping = param_config_dd[run]
            if grouping == 0:
                data_splits = [data]
            elif grouping == 1:
                data_splits = [patients, controls]
            elif grouping == 2:
                data_splits = [threat, safe]
            elif grouping == 3:
                data_splits = [patients_threat, patients_safe, controls_threat, controls_safe]
            for k in range(0, len(data_splits)):
                subjects = np.unique(data_splits[k].values[:, subject_id_col - 1])
                subject_no = subjects.shape[0]
                Tsubj = (np.zeros(subject_no, dtype=np.int))
                for subj in range(0, subject_no):
                    current_subj = subjects[subj]
                    Tsubj[subj] = sum(data_splits[k].values[:, subject_id_col - 1] == current_subj)
                max_trials = int(max(Tsubj))
                print(model_names[index], "\n")
                print("Number of Markov chains: ", chain_no, "\n")
                print("Number of cores used: ", core_no, "\n")
                print("Number of MCMC samples per chain: ", iter_no, "\n")
                print("Number of burn-in samples: ", warmup_no, "\n")
                print("Number of maximum trials per subject: ", max_trials, "\n")
                outcome = np.zeros((subject_no, max_trials))
                pressed = np.ones((subject_no, max_trials), dtype=np.int)
                cue = np.ones((subject_no, max_trials), dtype=np.int)
                for subj in range(0, subject_no):
                    current_subj = subjects[subj]
                    use_Trials = Tsubj[subj]
                    tmp = data_splits[k][data_splits[k].values[:, subject_id_col - 1] == current_subj]
                    outcome[subj, :use_Trials] = tmp.values[:, outcome_col - 1]
                    pressed[subj, :use_Trials] = tmp.values[:, keyPressed_col - 1]
                    cue[subj, :use_Trials] = tmp.values[:, cue_col - 1]
                data_list = {'N': subject_no, 'T': max_trials, 'Tsubj': Tsubj, 'outcome': outcome,
                             'pressed': pressed, 'cue': cue, 'param_config': param_config, 'grouping': grouping}
                genInitList = GNG_genInitList(model_names[index])
                # Fit the STAN model
                sm = StanModel(file=model_file)
                fit = sm.sampling(data=data_list, pars=POI, warmup=warmup_no, init=genInitList, iter=iter_no,
                                  chains=chain_no, thin=thin_no, control={'adapt_delta': adapt_delta,
                                                                          'max_treedepth': max_treedepth,
                                                                          'stepsize': stepsize})

