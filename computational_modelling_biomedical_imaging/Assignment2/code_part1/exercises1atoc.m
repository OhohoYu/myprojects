%
% Coursework 2 Part I - Exercise 1a-c
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
[hyp,p,ci,stats] = ttest2(group1, group2);
% generate design matrix 
oness = ones(size_G1,1);
zeross = zeros(size_G2,1);
design_mat = [oness,zeross;zeross,oness];
% dimension of col space of design matrix
dim = rank(design_mat);
% perpendicular projection operator Px corresponding to any col space C(X)
proj = design_mat*inv(design_mat'*design_mat)*design_mat';
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
betahat = inv(design_mat'*design_mat)*design_mat'*bothgroups;
% estimate the variance of the stochastic component ehat via
var = (ehat'*ehat)/48;
% covariance matrix of estimated model parameters
covar_mat = var*inv(design_mat'*design_mat);
% use covar_mat to determine st.dev of model parameters
stdev_params = sqrt(covar_mat(1,1));
% derive contrast vector for comparing group differences in means 
lambda = [1 -1]';
% reduced model X0 corresponding to null hypothesis 
X0 = ones(length(bothgroups),1);
% recalculate new betahat
betahat_new = inv(X0'*X0)*X0'*bothgroups;
% determine error via e = Y-X0B0
error_new = bothgroups-X0*betahat_new;
% additional error as result of placing constraint
additional_error = error_new - ehat;
% projection matrix for new design mat X0
newproj = X0*inv(X0'*X0)*X0';
% calculate v1, v2 (dof) for f-statistic
v1 = trace(proj-newproj);
v2 = trace(eye(length(bothgroups))-proj);
% estimate f-statistic of comparing X0 to X
SSRX0 = error_new'*error_new;
SSRX = ehat'*ehat;
fstat = ((SSRX0 - SSRX)/v1)/(SSRX/v2);
% determine t-statistic to test whether one group has higher mean
tstat = (lambda'*betahat)/(sqrt(lambda'*covar_mat*lambda));
% ground truth beta composed of ground truth means
beta_gt = [1 1.5]';
yhat = [1 1.5];
e_gtdeviation = bothgroups - design_mat*beta_gt;


