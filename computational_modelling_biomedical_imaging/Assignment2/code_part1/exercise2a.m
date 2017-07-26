%
% Coursework 2 Part I - Exercise 2a
%
% clear history
clear all; close all;
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
% perform one-sample t-test
[hyp,p,ci,stats] = ttest(group1, group2);