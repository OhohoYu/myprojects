% Antonio Remiro Azocar
% Coursework 2 Part II - Exercise 1
%
%% (a)
% clear history
clear all; close all;
% set random seed
rng(13)
% similar dataset as for Part I but smaller sample sizes
% group 1 size
size_G1 = 6;
% group 2 size
size_G2 = 8;
% means
mean_G1 = 1;
mean_G2 = 1.5;
% stochastic error
e1 = 0.25*randn(size_G1, 1);
e2 = 0.25*randn(size_G2, 1);
% generate datasets
group1 = mean_G1+e1;
group2 = mean_G2+e2;
% both groups; 
bothgroups = [group1; group2];
% perform two-sample t-test two determine t-statistic, p-value
[hyp,p,ci,stats] = ttest2(group1, group2);
% original ttest2 values
original_p = p;
original_tstat = stats.tstat;
original_meandiff = mean(group1)-mean(group2);
%% (b) compute the exact permutation-based p-value
%
% (i) construct an 1D array of size n = n1 (G1) + n2 (G2)
D = bothgroups';  
% (ii) construct valid permutations maintaining sample size
C1 = combnk(D, size_G1); 
C2 = zeros(length(C1),size_G2); 
tstat_array = zeros(1, length(C1)); % initialise array to store all t-statistics
for i=1:length(C1)
    C2(i,:) = setdiff(D, C1(i,:)); % no repeats (valid permutations)
    % (iii) compute t-statistics for all membership permutations
    [hyp,p,ci,stats] = ttest2(C1(i,:), C2(i,:));
    tstat_array(i) = stats.tstat;    
end
% construct empirical distribution of t-statistic
hist(tstat_array, 50)
% (iv) determine p-value by finding % permutations with tstat greater than that
% of original value
empirical_p = numel(tstat_array(tstat_array <= original_tstat))/...
                                numel(tstat_array);

%% (c)
% repeat (b) but rather than using t-statistic, use difference between
% means as test statistic
mean_diff_array = zeros(1, length(C1));
for i=1:length(C1)
   mean_diff = mean(C1(i,:)) - mean(C2(i,:));
   mean_diff_array(i) = mean_diff;
end
figure
hist(mean_diff_array, 50)
% empirical p-value calculated with mean diffs
c_empirical_p = numel(mean_diff_array(mean_diff_array <= ... 
                                      original_meandiff))/numel(mean_diff_array);

%% (d) compute approximate permutation-based p-value. 
tot_permutations = 1000;
tstat_array_d = zeros(1,tot_permutations);
all_permsi = zeros(tot_permutations, 6);
for i=1:tot_permutations
    % randperm generates random set of integer permutations
    % integers are indices for D
    perms = randperm(14);
    all_permsi(i,:) = perms(1:6);
    bothgroups = D(perms);
    group1 = bothgroups(1:6);
    group2 = bothgroups(7:14);
    [hyp,p,ci,stats] = ttest2(group1, group2);
    tstat_array_d(i) = stats.tstat;        
end                                  
figure
hist(tstat_array_d,50)
% empirical p-value calculated with mean diffs (1000 permutations only, t-stat)
d_empirical_p = numel(tstat_array_d(tstat_array_d <= ... 
                                      original_tstat))/numel(tstat_array_d);

%% (diii) p-value without duplicates                                  
%first carry out 1500 iterations, then remove duplicates and store only first 1000
%iterations
extended_permutations = 1500;
tstat_array_diii = zeros(1,tot_permutations);
all_perms = zeros(extended_permutations, 14);
for i = 1:extended_permutations
    % randperm generates random set of integer permutations
    % integers are indices for D
    perms = randperm(14);
    all_perms(i,:) = perms;
    bothgroups = D(perms);
    group1 = bothgroups(1:6);
    group2 = bothgroups(7:14);
    [hyp,p,ci,stats] = ttest2(group1, group2);
    tstat_array_diii(i) = stats.tstat;            
end                                  
% sort permutations so those with the same elements are identical
sorted_perms = sort(all_perms(:,1:6),2);
[~, uniqueids, ~]=unique(sorted_perms, 'rows');

% no duplicates

new_perms = sorted_perms(uniqueids,:);
new_perms = new_perms(1:tot_permutations, :);
tstat_array_diii = tstat_array_diii(uniqueids);
tstat_array_diii = tstat_array_diii(5:1005);

% empirical p-value calculated with mean diffs (all permutations, t-stat)
diii_empirical_p = numel(tstat_array_diii(tstat_array_diii  <= ... 
                                      original_tstat))/numel(tstat_array_diii);