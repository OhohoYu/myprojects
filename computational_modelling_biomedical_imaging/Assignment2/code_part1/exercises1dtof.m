%
% Coursework 2 Part I - Exercise 1d-f
% clear history
clear all; close all;
%
% 1d
%
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
% generate design matrix 
oness = ones(size_G1,1);
zeross = zeros(size_G2,1);
design_mat = [oness, oness,zeross;oness,zeross,oness];
% perpendicular projection operator Px corresponding to any col space C(X)
proj = design_mat*(pinv(design_mat'*design_mat))*design_mat';
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
% used derived formula to determine betahat, estimate to GLM model params
betahat = pinv(design_mat'*design_mat)*design_mat'*bothgroups;
% estimate the variance of the stochastic component ehat via
var = (ehat'*ehat)/48;
% covariance matrix of estimated model parameters
covar_mat = var*pinv(design_mat'*design_mat);
% derive contrast vector for comparing group differences in means 
lambda = [1 -1 0]';
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
%
% 1e
%
design_mat = [oness, oness;oness,zeross];
% perpendicular projection operator Px corresponding to any col space C(X)
proj = design_mat*(pinv(design_mat'*design_mat))*design_mat';
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
% used derived formula to determine betahat, estimate to GLM model params
betahat = pinv(design_mat'*design_mat)*design_mat'*bothgroups;
% estimate the variance of the stochastic component ehat via
var = (ehat'*ehat)/48;
% covariance matrix of estimated model parameters
covar_mat = var*pinv(design_mat'*design_mat);
% derive contrast vector for comparing group differences in means 
lambda = [0 1]';
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