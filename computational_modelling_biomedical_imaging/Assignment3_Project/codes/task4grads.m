%% TASK 4: FIT QUADRATIC/CUBIC MODELS (inhalation/exhalation)
close all

%% Add functions and data paths
addpath(genpath('data'))
addpath(genpath('for_students'))
load('SurrogateSignal.mat')

%% Load the first 100 control point registrations
n_train = 100;
regs = dir(['data' filesep 'reg_results' filesep 'Reg*.nii']);
CPs = cell(n_train,1);

for i=1:n_train
    CPs{i} = load_untouch_nii(regs(i).name);
end

% Extract the first 100 surrogate signal measurements
x = SurrogateSignal(1, 1:n_train)';
% Extract the first 100 gradients - the array is shifted by one e.g. x=2 is classed
% as an inhalation if the signal value increases from x=1 to x=2; 
grads = gradient(SurrogateSignal(1,2:n_train+1));
% grads = gradient(x);
% registrations are separated into those from end-inhalation/end=exhalation
inhalations = x(grads>0); % troughs of signal correspond to inhalation 
exhalations = x(grads<0); % peaks correspond to exhalation

n_cp = numel(CPs{1}.img(:,:,1,1,2)); % number of control points

n_train_inhale = length(inhalations); % number of inhalations
n_train_exhale = length(exhalations); % number of exhalations

%% INHALATION
%% Retrieve all the CP displacements
si_disp_mat_inhale = zeros(n_train_inhale, n_cp);
% Extract the CP displacements
for t=1:n_train_inhale
    si_disp_mat_inhale(t, :) = reshape(CPs{t}.img(:,:,1,1,2), 1, []);
end
Y_inhale= si_disp_mat_inhale;  % n_train x n_cp
%% Fit first-order (linear) model (again)
X_linear_inhale = [inhalations, ones(size(inhalations))];
W_linear_inhale = X_linear_inhale\Y_inhale;
Y_linear_fit_inhale = X_linear_inhale * W_linear_inhale;
%% Fit second-order (quadratic) model
X_quad_inhale = [inhalations.^2, inhalations, ones(n_train_inhale, 1)]; 
W_quad_inhale = X_quad_inhale\Y_inhale;
% Predicted surrogate signal
Y_quad_fit_inhale = X_quad_inhale*W_quad_inhale;
%% Fit third-order (cubic) model
X_cubic_inhale = [inhalations.^3, inhalations.^2, inhalations, ... 
                  ones(size(inhalations))];
W_cubic_inhale = X_cubic_inhale\Y_inhale;
Y_cubic_fit_inhale = X_cubic_inhale * W_cubic_inhale;
%% Compare residual fitting errors
Y_fits_inhale = {Y_linear_fit_inhale; Y_quad_fit_inhale; Y_cubic_fit_inhale};
Y_fit_names = {'Linear', 'Quadratic', 'Cubic'};
for fit_idx = 1:length(Y_fits_inhale)
     Y_fit_inhale = Y_fits_inhale{fit_idx};
     fprintf('=== %s Fit ===\n', Y_fit_names{fit_idx})
     errs = (Y_fit_inhale - Y_inhale).^2;
     rss = sum(errs, 1);
     total_rss = sum(rss);
     mean_rss = mean(rss);
     disp('Total residual sum of squares:')
     disp(total_rss)
     disp('Mean residual sum of squares:')
     disp(mean_rss);
end
%% Plot for CP 30, 20
ind = sub2ind(size(CPs{1}.img), 30, 20);
figure();
plot(inhalations, Y_inhale(:, ind), 'LineStyle', ':', 'Marker', 'x')
hold on
for fit_idx = 1:length(Y_fits_inhale)
    plot(inhalations, Y_fit_inhale(:, ind))
end
legend('data', 'linear', 'quadratic', 'cubic')

%% EXHALATION
%% Retrieve all the CP displacements
si_disp_mat_exhale = zeros(n_train_exhale, n_cp);
% Extract the CP displacements
for t=1:n_train_exhale
    si_disp_mat_exhale(t, :) = reshape(CPs{t}.img(:,:,1,1,2), 1, []);
end
Y_exhale= si_disp_mat_exhale;  % n_train x n_cp
%% Fit first-order (linear) model (again)
X_linear_exhale = [exhalations, ones(size(exhalations))];
W_linear_exhale = X_linear_exhale\Y_exhale;
Y_linear_fit_exhale = X_linear_exhale * W_linear_exhale;
%% Fit second-order (quadratic) model
X_quad_exhale = [exhalations.^2, exhalations, ones(n_train_exhale, 1)]; 
W_quad_exhale = X_quad_exhale\Y_exhale;
% Predicted surrogate signal
Y_quad_fit_exhale = X_quad_exhale*W_quad_exhale;
%% Fit third-order (cubic) model
X_cubic_exhale = [exhalations.^3, exhalations.^2, exhalations, ones(size(exhalations))];
W_cubic_exhale = X_cubic_exhale\Y_exhale;
Y_cubic_fit_exhale = X_cubic_exhale * W_cubic_exhale;
%% Compare residual fitting errors
Y_fits_exhale = {Y_linear_fit_exhale; Y_quad_fit_exhale; Y_cubic_fit_exhale};
Y_fit_names = {'Linear', 'Quadratic', 'Cubic'};
for fit_idx = 1:length(Y_fits_exhale)
     Y_fit_exhale = Y_fits_exhale{fit_idx};
     fprintf('=== %s Fit ===\n', Y_fit_names{fit_idx})
     errs = (Y_fit_exhale - Y_exhale).^2;
     rss = sum(errs, 1);
     total_rss = sum(rss);
     mean_rss = mean(rss);
     disp('Total residual sum of squares:')
     disp(total_rss)
     disp('Mean residual sum of squares:')
     disp(mean_rss);
end
%% Plot for CP 30, 20
ind = sub2ind(size(CPs{1}.img), 30, 20);
figure();
plot(exhalations, Y_exhale(:, ind), 'LineStyle', ':', 'Marker', 'x')
hold on
for fit_idx = 1:length(Y_fits_exhale)
    plot(exhalations, Y_fit_exhale(:, ind))
end
legend('data', 'linear', 'quadratic', 'cubic')