clear all;
rng(77)

% generate dataset like that of Q2

ntot = 600; % number of data points
dim = 10; % number of dimensions
ntrain = 100; % number of training examples 
ntest = ntot - ntrain; % number of test examples

a = -6:1:3; % range of powers of ten used
for i=1:length(a)
    gamma(i) = 10^a(i);  % gamma generation
end
gamma_size = numel(gamma);

runs = 200; 

train_errors = zeros(1, gamma_size); % keeps track of training errors
test_errors = zeros(1, gamma_size); % keeps track of test errors


for i=1:runs
    w = randn(1,dim); % pick random value for w from st normal dist.
    x = randn(ntot,dim); % x samples drawn from st normal dist.
    n = randn(ntot,1); % noise drawn from st normal dist.
    y = x*w' + n; % noisy random dataset generated
    % original train/test split
    x_train = x(1:ntrain,:); 
    y_train = y(1:ntrain);
    x_test = x(ntrain+1:end,:);
    y_test= y(ntrain+1:end);
    for j=1:gamma_size
       % obtain alpha parameters for dual ridge regression
       alpha = mldivide((x_train*x_train' + gamma(j)...
                                *ntrain*eye(ntrain)),y_train);
       w_learned = x_train' * alpha; % learned weights from train
       % predictions
       y_test_predict = x_test * w_learned;
       y_train_predict = x_train * w_learned;
       % errors
       MSE_train = (norm(y_train_predict - y_train)^2) /ntrain;
       MSE_test = (norm(y_test_predict - y_test)^2) /ntest;
       % tracks errors over different parameter values
       train_errors(j) = train_errors(j) + MSE_train;
       test_errors(j) = test_errors(j) + MSE_test; 
       
    end
    disp(test_errors)
end    

% computing average values
avg_train_errors = train_errors./runs;
avg_test_errors = test_errors./runs;

% plots 
figure
semilogx(gamma,avg_train_errors, '-.or')
hold on 
semilogx(gamma, avg_test_errors, ':+k')
hold off
h = legend('training', 'test','Location','north');
set(h, 'FontSize', 14);