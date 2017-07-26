clear all;

% generate dataset like that of Q2

ntot = 600; % number of data points
dim = 10; % number of dimensions
ntrain_original = 100; % number of training examples (original)
ntest = ntot - ntrain_original; % number of test examples

% 80/20 train/validation split
ntrain_new = floor(0.8 .* ntrain_original); % ntrain2 is the new training set size
nval = ntrain_original - ntrain_new;

a = -6:1:3; % range of powers of ten used
for i=1:length(a)
    gamma(i) = 10^a(i);  % gamma generation
end
gamma_size = numel(gamma);

runs = 200; 

train_new_errors = zeros(1, gamma_size); % keeps track of training errors
val_errors = zeros(1, gamma_size); % keeps track of validation errors
test_errors = zeros(1, gamma_size); % keeps track of test errors
tot_min_gamma = [];  % keeps track of gamma values. 
tot_definitive_test_err = []; % keeps track of definitive test error. 

for i=1:runs
    min_val_err = 10^8; % dummy variable for minimum val error calculation
    w = randn(1,dim); % pick random value for w from st normal dist.
    x = randn(ntot,dim); % x samples drawn from st normal dist.
    n = randn(ntot,1); % noise drawn from st normal dist.
    y = x*w' + n; % noisy random dataset generated
    % original train/test split
    x_train_original = x(1:ntrain_original,:); 
    y_train_original = y(1:ntrain_original);
    x_test = x(ntrain_original+1:end,:);
    y_test= y(ntrain_original+1:end);
    % split original train into train and val
    x_train_new = x(1:ntrain_new, :);
    y_train_new = y(1:ntrain_new, :);
    x_val = x(ntrain_new+1:ntrain_original, :);
    y_val = y(ntrain_new+1:ntrain_original, :);
    for j=1:gamma_size
       % obtain alpha parameters for dual ridge regression
       alpha = mldivide((x_train_new*x_train_new' + gamma(j)...
                                *ntrain_new*eye(ntrain_new)),y_train_new);
       w_learned = x_train_new' * alpha; % learned weights from train
       % predictions
       y_val_predict = x_val * w_learned; 
       y_train_new_predict = x_train_new * w_learned;
       y_test_predict = x_test * w_learned;
       % errors
       MSE_val = (norm(y_val_predict - y_val)^2) / nval;
       MSE_train_new = (norm(y_train_new_predict - y_train_new)^2)/ntrain_new;
       MSE_test = (norm(y_test_predict - y_test)^2) /ntest;
       % tracks errors over different parameter values
       train_new_errors(:,j) = train_new_errors(j) + MSE_train_new;
       val_errors(:,j) = val_errors(j) + MSE_val; 
       test_errors(:,j) = test_errors(j) + MSE_test; 
       % keeps track of optimal gamma for a particular run 
       if MSE_val < min_val_err;
            min_val_err=MSE_val;
            min_gamma = gamma(j);
       end   
    end
    % append to gamma tracker
    tot_min_gamma = [tot_min_gamma min_gamma]; 
    % use both train and validation sets to train weights for test error
    alpha_new = mldivide((x_train_original*x_train_original' + min_gamma...
                           *ntrain_original*eye(ntrain_original)),y_train_original);
    w_learned_new = x_train_original' * alpha_new; % learned weights from train
    % test predictions using optimal gamma
    y_test_predict_new = x_test * w_learned_new;
    % keeps track of final test errors
    definitive_test_error = (norm(y_test_predict_new - y_test)^2) / ntest;
    [tot_definitive_test_err] = [tot_definitive_test_err definitive_test_error];
end    

% computing average values
avg_train_new_errors = train_new_errors./runs;
avg_val_errors = val_errors./runs;
avg_test_errors = test_errors./runs;
avg_gamma = sum(tot_min_gamma)/runs;
avg_definitive_test_error = sum(tot_definitive_test_err)/runs;
std_test_error = std(tot_definitive_test_err);

% plots 
figure
semilogx(gamma,avg_train_new_errors, '-.or')
hold on 
semilogx(gamma, avg_val_errors, '--xb')
semilogx(gamma, avg_test_errors, ':+k')
hold off
h = legend('training', 'validation','test','Location','northwest');
set(h, 'FontSize', 14);