% Exercise 1c
%
clear all
%
dim = 1; % dimensions
ntot = 510;
ntrain = 10; 
ntest = 500;
%
w = randn(1,dim); % pick random value for w from st normal dist.
x = randn(ntot,dim); % 510 samples drawn from st normal dist. 
n = randn(ntot,1); % noise generated from st normal dist.
y = x*w' + n; % noisy random dataset generated
% 
% split dataset into training set size 10, test set size 500
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
               -(2*y_train'*x_train*train_weights)+(y_train'*y_train))/ntrain;
MSE_test = ((train_weights'*x_test'*x_test*train_weights)...
               -(2*y_test'*x_test*train_weights)+(y_test'*y_test))/ntest;

