%function [predCPs, ] = Task5()
%% TASK 5: EVALUATE LINEAR MODEL
%clear all
close all

%% Add functions and data paths
addpath(genpath('data'))
addpath(genpath('for_students'))
load('SurrogateSignal.mat')

%% Training/Test split
n_train = 100;
n_test = 200;

%% Divide surrogate signals into training & test set
xTrain = SurrogateSignal(1, 1:n_train)';
xTest = SurrogateSignal(1, n_train+1:end)';

%% Load training CPs
regs = dir('data/reg_results/Reg*.nii');
CPs = cell(n_train,1);

for i=1:n_train
    CPs{i} = load_untouch_nii(regs(i).name);
end

% Build displacement matrix
[img_h, img_w] = size(CPs{1}.img(:,:,1,1,2));
n_cp = numel(CPs{1}.img(:,:,1,1,2));
si_disp_mat = zeros(n_train, n_cp, 2);
% Extract the CP displacements (x, y) coords
for t=1:n_train
    si_disp_mat(t, :, :) = reshape(CPs{t}.img(:,:,1,1,:), 1, [], 2);
end

% Training Design matrix
Xtrain = [xTrain, ones(n_train, 1)];  % n_train x 2

%% Fit X and Y displacements (Linear model)
Wx = Xtrain\si_disp_mat(:, :, 1);
Wy = Xtrain\si_disp_mat(:, :, 2);

% Mean residual error X
errs = (Xtrain*Wx - si_disp_mat(:, :, 1)).^2;
rss = sum(errs, 1);
fprintf('Fitted model for x disp MSE: %f\n', mean(rss));

% Mean residual error X
errs = (Xtrain*Wy - si_disp_mat(:, :, 2)).^2;
rss = sum(errs, 1);
fprintf('Fitted model for y disp MSE: %f\n', mean(rss));


%% Evaluate Model
% Testing design matrix
Xtest = [xTest, ones(n_test, 1)];  % n_test x 2

% Compute displacement on x coord
disp_x_pred = Xtest*Wx;
disp_x_pred = reshape(disp_x_pred, n_test, img_h, img_w, 1, 1, 1);

% Compute displacement on y coord
disp_y_pred = Xtest*Wy;
disp_y_pred = reshape(disp_y_pred, n_test, img_h, img_w, 1, 1, 1);

% Concatenate x and y displacements 6-D tensor
disp_pred = cat(6, disp_x_pred, disp_y_pred);

%% Save predicted CPs results to nifti format
predCPs = cell(n_test, 1);
for t=1:n_test
    % Copy header from 1st registration
    predCPs{t} = CPs{1};
    % Replace img with predicted displacements
    predCPs{t}.img = reshape(disp_pred(t,:,:,:,:,:), img_h, img_w, 1, 1, 2);
end










