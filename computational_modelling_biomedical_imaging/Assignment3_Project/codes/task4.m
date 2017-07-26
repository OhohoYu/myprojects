%% TASK 3: FIT QUADRATIC/CUBIC MODELS
close all
clear all

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

%% Retrieve all the CP displacements
n_cp = numel(CPs{1}.img(:,:,1,1,2));
si_disp_mat = zeros(n_train, n_cp);

% Extract the CP displacements
for t=1:n_train
    si_disp_mat(t, :) = reshape(CPs{t}.img(:,:,1,1,2), 1, []);
end

Y = si_disp_mat;  % n_train x n_cp

%% Fit first-order (linear) model (again)
X_linear = [x, ones(size(x))];
W_linear = X_linear\Y;
Y_linear_fit = X_linear * W_linear;

%% Fit second-order (quadratic) model
X_quad = [x.^2, x, ones(n_train, 1)]; 
W_quad = X_quad\Y;
% Predicted surrogate signal
Y_quad_fit = X_quad*W_quad;

%% Fit third-order (cubic) model
X_cubic = [x.^3, x.^2, x, ones(size(x))];
W_cubic = X_cubic\Y;
Y_cubic_fit = X_cubic * W_cubic;

%% Compare residual fitting errors
Y_fits = {Y_linear_fit; Y_quad_fit; Y_cubic_fit};
Y_fit_names = {'Linear', 'Quadratic', 'Cubic'};

for fit_idx = 1:length(Y_fits)
    Y_fit = Y_fits{fit_idx};
    fprintf('=== %s Fit ===\n', Y_fit_names{fit_idx})
    errs = (Y_fit - Y).^2;
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


% figure(1);
% plot(x, Y(:, ind), 'LineStyle', ':', 'Marker', 'x')
% hold on
% for fit_idx = 1:length(Y_fits)
%     plot(x, Y_fit(:, ind))
% end
% legend('data', 'linear', 'quadratic', 'cubic')

figure(2)
plot(x, Y(:, ind), 'LineStyle', ':', 'Marker', 'x')
hold on
plot(x, Y_fit(:, ind))











