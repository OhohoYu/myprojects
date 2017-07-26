% Exercise 2a
%
clear all
%
dim = 10; % dimensions
ntot = 600;
ntrain = 100; 
ntest = 500;
%
w = randn(1,dim); % pick random value for w from st normal dist.
x = randn(ntot,dim); % 600 samples drawn from st normal dist. 
n = randn(ntot,1); % noise generated from st normal dist.
y = x*w' + n; % noisy random dataset generated
% 
% split dataset into training set size 100, test set size 500
x_train = x(1:ntrain,:);
y_train = y(1:ntrain);
x_test = x(ntrain+1:end,:);
y_test = y(ntrain+1:end);