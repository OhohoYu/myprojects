%% Task 5: AIC/BIC
addpath(genpath('for_students'))
addpath(genpath('data'))
clear
close all
% Extract train and test surrogate signal (xtrain, xtest) and train and
% test target (SI, AP displacements)
% Fit AP and SI models on the training data
% Use fitted models to predict the test data
% Compute RSS and hence AIC/BIC using the predictions
% load('ModelWeights.mat')
% dir_idx = 1; % AP displacement
dir_idx = 2; % SI displacement


%% Get true signal for 200 test images
n_train = 100; n_test = 200;
regs = dir(['data' filesep 'reg_results' filesep 'Reg*.nii']);
CPs = cell(n_test,1);

for i=1:n_test+n_train
    CPs{i} = load_untouch_nii(regs(i).name);
end

train_CPs = CPs{1:n_train};
test_CPs = CPs{n_train+1:end};
n_cp = numel(CPs{1}.img(:,:,1,1,2));

YTrain = zeros(n_train, n_cp);
YTest = zeros(n_test, n_cp);

% Extract the CP displacements
for t=1:n_train
    YTrain(t, :) = reshape(CPs{t}.img(:,:,1,1,dir_idx), 1, []);
end

for t=1:n_test
    YTest(t, :) = reshape(CPs{t+n_train}.img(:,:,1,1,dir_idx), 1, []);
end

%% Extract surrogate signal
load('SurrogateSignal.mat')
xTrain = SurrogateSignal(1, 1:100)';
xTest = SurrogateSignal(1, 101:300)';

%% Training

% Linear 
X_linear = [xTrain, ones(size(xTrain))];
W_linear = X_linear\YTrain;

% Quadratic
X_quad = [xTrain.^2, xTrain, ones(size(xTrain))]; 
W_quad = X_quad\YTrain;

% Cubic
X_cubic = [xTrain.^3, xTrain.^2, xTrain, ones(size(xTrain))];
W_cubic = X_cubic\YTrain;


%% Testing

% Linear
X_linear = [xTest, ones(size(xTest))];
Y_linear_fit = X_linear * W_linear;

% Fit second-order (quadratic) model
X_quad = [xTest.^2, xTest, ones(size(xTest))]; 
Y_quad_fit = X_quad * W_quad;

% Fit third-order (cubic) model
X_cubic = [xTest.^3, xTest.^2, xTest, ones(size(xTest))];
Y_cubic_fit = X_cubic * W_cubic;


%% Compute RSS; Make Histogram; Combine into grid
Y_fits = {Y_linear_fit; Y_quad_fit; Y_cubic_fit};
Y_fit_names = {'Linear', 'Quadratic', 'Cubic'};
rss_grid = zeros(48,48,3);
rss_arr = cell(3,1);

for i=1:3
    rss_arr{i} = sum((YTest - Y_fits{i}).^2, 1);
    mean_rss = mean(rss_arr{i});
    rss_grid(:,:,i) = reshape(rss_arr{i}, 48, 48);
    
    subplot(3,3,i)
%     subplot(3,2,2*i-1)
    histogram(log10(rss_arr{i}))
    title(sprintf('%s\nMean RSS %g',Y_fit_names{i}, mean_rss))
    line(repmat(log10(mean_rss), [1 2]), get(gca, 'YLim'))
    xlabel('Log_{10} RSS')
    
    set(gca, 'FontSize', 8)
end

% Show RSS parameter map
for i=1:3
    subplot(3,3,[3+i 6+i])
%     subplot(3,2,2*i)
    imagesc(rss_grid(:,:,i)')
    c=colorbar
    c.Location = 'southoutside';
    axis square; 
    axis off;
    set(gca, 'XLim', [0 48], 'YLim', [0 48], 'FontSize', 8);
end

set(gca, 'FontSize', 8)
set(gcf, 'PaperPosition', [0 0 15 12], 'PaperSize', [15 12])
print(sprintf('./fig/ic/rss_hist_map-%g.png', dir_idx), '-dpng')

%% Compute AIC/BIC
% N: number of model parameters
% K: number of data points
n_params = {2; 3; 4};
K = n_test;
aic = zeros(48*48, 3);

figure();
for i=1:3
    N = n_params{i};
    RSS = rss_arr{i};
    aic(:,i) = 2*N + K.*log(RSS./K);
    bic(:,i) = N*log(K) + K.*log(RSS./K);
    
    disp(sprintf('=== %s Fit ===', Y_fit_names{i}))
    disp('Mean AIC')
    disp(mean(aic(:,i)))
    disp('Mean BIC')
    disp(mean(bic(:,i)))
    
    subplot(2,3,i)
    imagesc(reshape(aic(:,i), 48, 48)')
    title(sprintf('%s AIC', Y_fit_names{i}))
    colorbar
    axis square; set(gca, 'XLim', [0 48], 'YLim', [0 48], 'FontSize', 8);
    
    subplot(2,3,3+i)
    imagesc(reshape(bic(:,i), 48, 48)')
    title(sprintf('%s BIC', Y_fit_names{i}))
    colorbar
    axis square; set(gca, 'XLim', [0 48], 'YLim', [0 48], 'FontSize', 8);
end

set(gcf, 'PaperPosition', [0 0 16 9], 'PaperSize', [16 9])
print(sprintf('./fig/ic/aic_bic_map-%g.png', dir_idx), '-dpng')


%% Show AIC/BIC grid
aic_grid = reshape(aic, 48, 48, 3);
bic_grid = reshape(bic, 48, 48, 3);

[~, aic_min] = min(aic_grid, [], 3);
[~, bic_min] = min(bic_grid, [], 3);

figure('PaperPosition', [0 0 10 10], 'PaperSize', [10 10]);
% subplot(2,1,1)
imagesc(aic_min')
% title('AIC')
axis square; axis off;
set(gca, 'XLim', [0 48], 'YLim', [0 48], 'FontSize', 8);
print(sprintf('./fig/ic/AIC_model_select_map-%g.png', dir_idx), '-dpng')

figure('PaperPosition', [0 0 10 10], 'PaperSize', [10 10]);
% subplot(2,1,2)
imagesc(bic_min')
% title('BIC')
axis square; axis off;
set(gca, 'XLim', [0 48], 'YLim', [0 48], 'FontSize', 8);
print(sprintf('./fig/ic/BIC_model_select_map-%g.png', dir_idx), '-dpng')


set(gcf, 'PaperPosition', [0 0 9 16], 'PaperSize', [9 16])
print(sprintf('./fig/ic/model_select_map-%g.png', dir_idx), '-dpng')

% Blue: linear model
% Teal: quadratic model
% Yellow: cubic model
% APPLY MASK?