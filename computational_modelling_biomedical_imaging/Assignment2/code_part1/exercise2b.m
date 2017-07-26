%
% Coursework 2 Part I - Exercise 2b
%
% clear history
clear all; close all;
% set random seed
rng(3)
% group 1 size
size_G1 = 25;
% group 2 size
size_G2 = 25;
% means
mean_G1 = 1;
mean_G2 = 1.5;
% stochastic error
e1 = 0.25*randn(size_G1, 1);
e2 = 0.25*randn(size_G1, 1);
% generate datasets
group1 = mean_G1+e1;
group2 = mean_G2+e2;
% both groups; 
bothgroups = [group1; group2];
% perform two-sample t-test
[hyp,p,ci,stats] = ttest(group1, group2);
% generate design matrix 
oness = ones(size_G1,1);
zeross = zeros(size_G2,1);
identity25 = eye(size_G1); twoidentities = [identity25; identity25];
firsttwocolumns = [oness,oness;oness,zeross];
design_mat = [firsttwocolumns, twoidentities];
% dimension of col space of design matrix
dim = rank(design_mat);
% perpendicular projection operator Px corresponding to any col space C(X)
proj = design_mat*pinv(design_mat'*design_mat)*design_mat';
% is proj idempotent? we check and it is
projsq = proj^2; 
% is proj symmetric? we check and it is 
projt = proj';
% use proj to determine projy, the projection of y into C(x)
projy = proj*bothgroups;
% compute Rx = I - Px 
Rx = eye(size(bothgroups,1))- proj;
% is Rx idempotent? we check and it is
Rxsq = Rx^2;
% is Rx symmetric? we check and it is 
Rxt = Rx';
% determine \hat{e}, the projection of Y into he error space
ehat = Rx*bothgroups; 
% determine the angle between ehat and Yhat. is it 90 degrees?
theta = acos(dot(ehat,projy));
% used derived formula to determine betahat, estimate to GLM model params
betahat = pinv(design_mat'*design_mat)*design_mat'*bothgroups;
% estimate the variance of the stochastic component ehat via
var = (ehat'*ehat)/24;
% covariance matrix of estimated model parameters
covar_mat = var*pinv(design_mat'*design_mat);
% use covar_mat to determine st.dev of model parameters
stdev_params = sqrt(covar_mat(1,1));
% derive contrast vector for comparing group differences in means 
lambda = [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]';
% determine t-statistic to test whether one group has higher mean
tstat = (lambda'*betahat)/(sqrt(lambda'*covar_mat*lambda));
