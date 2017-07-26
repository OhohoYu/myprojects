% Antonio Remiro Azocar
% Coursework 2 Part II - Exercise 2
% 
% 
clear history, clear all
%% data preprocessing

% load files
CPA_nos = [4,5,6,7,8,9,10,11]; % identifier for CPA file names
PPA_nos = [3,6,9,10,13,14,15,16]; % identifier for PPA file names
n_subjects = 8;
% load fractional anisotropy maps for group 1 (CPA)
% group 1 has 8 subjects 
% CPA data is merged
CPAdata = zeros(64000,n_subjects);
j = 0;
for i=drange(CPA_nos)   
    filename = sprintf('CPA%d_diffeo_fa.img',i);
    fid = fopen(filename, 'r', 'l'); % little-endian
    CPA_sub_data = fread(fid, 'float'); % 16-bit floating point
    j=j+1;
    CPAdata(:,j) = CPA_sub_data;    
end
% load fractional anisotropy maps for group 2 (PPA)
% group 2 has 8 subjects 
% PPA data is merged 
PPAdata = zeros(64000, n_subjects);
j = 0;
for i=drange(PPA_nos)   
    filename = sprintf('PPA%d_diffeo_fa.img',i);
    fid = fopen(filename, 'r', 'l'); % little-endian
    PPA_sub_data = fread(fid, 'float'); % 16-bit floating point
    j=j+1;
    PPAdata(:,j) = PPA_sub_data;    
end
% wm_mask.img is an additional binary volume defining the ROI for 
% statistical analysis 
fid = fopen('wm_mask.img', 'r', 'l');
mask = fread(fid, 'float');
voxels = [CPAdata PPAdata];

%% (a) 

% Use a GLM of choice: we use the GLM of part (1c) Y = X1b1 + X2b2 + e.

% generate design matrix 
oness = ones(n_subjects,1);
zeross = zeros(n_subjects,1);
design_mat = [oness,zeross;zeross,oness];
% contrast matrix
lambda = [1 -1]';
% calculate \beta
betahat = voxels*(((design_mat'*design_mat)\design_mat')');
% compute error vector 
ehat = voxels-(betahat*design_mat');
% covariance matrix of estimated model parameters
var = sum((ehat.*ehat), 2)./ 14;
t_stat = (betahat*lambda)./sqrt(var.*(lambda'*(inv(design_mat'*...
                                                         design_mat))*lambda));
% compute for specific region of interest
ROItstat = t_stat.*mask;
% compute maximum t-statistic. 
maxtstat = max(ROItstat);

%% (b)
size_g1 = 8;
size_g2 = 8;
indices = 1:(size_g1+size_g2);
% indicies for group1/group2 assignment like in (1b)
C1 = combnk(indices, size_g1);
C2 = zeros(length(C1),size_g2); 
% group index permutation
for i=1:length(C1)
    C2(i,:) = setdiff(indices, C1(i,:)); % no repeats (valid permutations)
end
bothgroups = [C1 C2];
maxtstatlist = zeros((length(C1)),1);

% ignore values which are not in ROI 
voxels = voxels.*mask;

for i=1:length(C1)
    i
    mix = voxels(:,bothgroups(i,:));
    betahat = mix*((design_mat'*design_mat)\design_mat')';
    ehat = mix-betahat*design_mat';
    % find std of error
    var = sum((ehat.*ehat), 2)./ 14;
    tstat = (betahat*lambda)./sqrt(var.*(lambda'*(inv(design_mat'*...
                                                         design_mat))*lambda));
    maxtstatlist(i,1) = max(tstat); 
end
figure
hist(maxtstatlist,50)

%% (c)

empirical_p = numel(maxtstatlist(maxtstatlist >= ... 
                                    maxtstat))/numel(maxtstatlist);

%% (d)
% determine maximum t-statistic threshold corresponding to p-value of 5% 
prctile(maxtstatlist, 95)









