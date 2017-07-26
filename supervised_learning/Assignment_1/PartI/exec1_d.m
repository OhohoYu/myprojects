% Exercise 1d
%
clear all
%
dim = 1; % dimensions
trials = 200;
ntot = 600;
ntrain = 100;
ntest = 500;
tot_MSE_train = [];  % keeps track of training errors
tot_MSE_test = []; % keeps track of test errors
%
for i=1:trials
    w = randn(1,dim); % pick random value for w from st normal dist.
    x = randn(ntot,dim); % samples drawn from st normal dist. 
    n = randn(ntot,1); % noise generated from st normal dist.
    y = x*w' + n; % noisy random dataset generated
    % 
    % split dataset into training set, test set. 
    x_train = x(1:ntrain,:);
    y_train = y(1:ntrain);
    x_test = x(ntrain+1:end,:);
    y_test = y(ntrain+1:end);
    %
    % using eqn 5, estimate weights based on training set
    train_weights = mldivide((x_train'*x_train), x_train'*y_train);
    %
    % using eqn 1, calculate predictive values
    y_train_predict = x_train*train_weights;
    y_test_predict = x_test*train_weights;
    %
    % using eqn 3, compute MSE on train and test sets
    MSE_train = ((train_weights'*x_train'*x_train*train_weights)...
                -(2*y_train'*x_train*train_weights)+(y_train'*...
                y_train))/ntrain;
    MSE_test = ((train_weights'*x_test'*x_test*train_weights)...
               -(2*y_test'*x_test*train_weights)+(y_test'...
               *y_test))/ntest;
    % append errors for each trial to total       
    tot_MSE_train = [tot_MSE_train MSE_train];
    tot_MSE_test = [tot_MSE_test MSE_test];
end
%
% calculate averages and st.dev
avg_MSE_train = sum(tot_MSE_train)/trials;
stdev_MSE_train = std(tot_MSE_train);
avg_MSE_test = sum(tot_MSE_test)/trials;
stdev_MSE_test = std(tot_MSE_test);  