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

dir_idx = 2; % AP displacement
% dir_idy = 2; % SI displacement


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

%% Extract surrogate signal 1
load('SurrogateSignal.mat')
xTrain = SurrogateSignal(1, 1:100)';
xTest = SurrogateSignal(1, 101:300)';

%% Extract surrogate signal 2
load('SurrogateSignal2.mat')
xTrain2 = SurrogateSignal2(1,2:100)';
xTest2 = SurrogateSignal2(1,101:300)';

% gradients
gradstrain = gradient(SurrogateSignal(1,2:n_train+1))';
gradstest = gradient(SurrogateSignal(1,n_train+2:end))';

% respiratory phase
% RESPIRATORY PHASE PREPROCESSING

% Extract the first 100 surrogate signal measurements
x = SurrogateSignal(1, 1:299)';
% Extract the first 100 gradients - the array is shifted by one e.g. x=2 is classed
% as an inhalation if the signal value increases from x=1 to x=2; 
grads = gradient(SurrogateSignal(1,2:300));
respiratory_phase = ones(1,299);
respiratory_phase(1)=0; % heuristically seen
end_inhales_ids = [];
end_exhales_ids = 1;
for i=2:299    
    if ((grads(i)<0) && (grads(i-1)>0))       
        end_inhales_ids = [end_inhales_ids i];
    end
    % calculate average 
    if ((grads(i)>0) && (grads(i-1)<0))
        end_exhales_ids = [end_exhales_ids i];
        respiratory_phase(i)=0; % end exhales are zero
    end    
end

percentage_container = zeros(1,25);
for j=1:25
    end_inhale = end_inhales_ids(j);
    lower_interval = end_inhale - end_exhales_ids(j);
    tot_interval = end_exhales_ids(j+1) - end_exhales_ids(j);
    percentage = (lower_interval/tot_interval);
    percentage_container(j)=percentage;
end
average_percent = mean(percentage_container);

for i=2:299
    if ((grads(i)<0) && (grads(i-1)>0))       
        respiratory_phase(i)=average_percent;
    end    
end    

% empirically seen
respiratory_phase(234)=0;
end_exhales_ids_a = end_exhales_ids(1:26);
end_exhales_ids_a = [end_exhales_ids_a 234];
end_exhales_ids_b = end_exhales_ids(27:end);
end_exhales_ids = [end_exhales_ids_a end_exhales_ids_b];

for i=1:length(end_exhales_ids)-1    
    for j=2:299
        if (j>end_exhales_ids(i)) && (j<end_inhales_ids(i))
            li = j - end_exhales_ids(i);
            ti = end_inhales_ids(i)-end_exhales_ids(i);
            respiratory_phase(j)=(li/ti)*average_percent;
        elseif (j<end_exhales_ids(i+1)) && (j>end_inhales_ids(i))
            ui = j-end_inhales_ids(i);
            ti = end_exhales_ids(i+1)-end_inhales_ids(i);
            respiratory_phase(j)=average_percent+(ui*(1-average_percent)/ti);            
        end    
    end
end
% again seen heuristically
respiratory_phase(end)=0.11;

RPTrain=respiratory_phase(1, 1:100)';
RPTest = respiratory_phase(1, 101:299)';

% %% Training

YTrain_new = YTrain;
% Model1
X_linear = [xTrain, ones(size(xTrain))];
W_linear = X_linear\YTrain_new;

% Model2
X_quad = [xTrain.^2, xTrain, ones(size(xTrain))]; 
W_quad = X_quad\YTrain_new;

% Model3
X_cubic = [xTrain.^3, xTrain.^2, xTrain, ones(size(xTrain))];
W_cubic = X_cubic\YTrain_new;

% Model 5
X5 = [xTrain.^2, xTrain, gradstrain, ones(size(xTrain))];
W5 = X5\YTrain_new;

% Model 
YTrain_new = YTrain(2:100,:);
X8 = [xTrain(2:100), xTrain2, ones(size(xTrain2))];
W8 = X8\YTrain_new;

% Testing

% Linear
X_linear = [xTest, ones(size(xTest))];
Y_linear_fit = X_linear * W_linear;

% Fit second-order (quadratic) model
X_quad = [xTest.^2, xTest, ones(size(xTest))]; 
Y_quad_fit = X_quad * W_quad;

% Fit third-order (cubic) model
X_cubic = [xTest.^3, xTest.^2, xTest, ones(size(xTest))];
Y_cubic_fit = X_cubic * W_cubic;

% Fit Model 5
xTestnew = xTest(1:199);
YTestnew = YTest(1:199,:);
X5 = [xTestnew.^2, xTestnew, gradstest, ones(199,1)];
Y5fit = X5 * W5;

% Fit Model 8
% Model 
X8 = [xTest, xTest2, ones(size(xTest2))];
Y8fit = X8*W8;

%% Compute RSS; Make Histogram; Combine into grid
Y_fits = {Y_linear_fit, Y_quad_fit, Y_cubic_fit, Y5fit, Y8fit};
Y_fit_names = {'Model 1', 'Model2', 'Model 3', 'Model 5', 'Model 8'};
rss_grid = zeros(48,48,5);
rss_arr = cell(5,1);

for i=1:5
    if i==4
    
        rss_arr{i} = sum((YTestnew - Y_fits{i}).^2, 1);
    else
        rss_arr{i} = sum((YTest - Y_fits{i}).^2, 1);
    end
    mean_rss = mean(rss_arr{i});
    rss_grid(:,:,i) = reshape(rss_arr{i}, 48, 48); 
    subplot(5,5,i)
% subplot(3,2,2*i-1)
    histogram(log10(rss_arr{i}))
    title(sprintf('%s\nMean RSS %g',Y_fit_names{i}, mean_rss))
    line(repmat(log10(mean_rss), [1 2]), get(gca, 'YLim'))
    xlabel('Log_{10} RSS')
    
    set(gca, 'FontSize', 8)
end
% Show RSS parameter map
for i=1:5
    subplot(5,5,[5+i 5+i])
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

%% Compute AIC/BIC
% N: number of model parameters
% K: number of data points
n_params = {2; 3; 4; 4; 3};
K = n_test;
aic = zeros(48*48, 5);
bic = zeros(48*48, 5);
figure();
for i=1:5
    N = n_params{i};
    RSS = rss_arr{i};
    if i==4
        K=199; 
    else
        K=200;
    end    
    aic(:,i) = 2*N + K.*log(RSS./K);
    bic(:,i) = N*log(K) + K.*log(RSS./K); 
    disp(sprintf('=== %s Fit ===', Y_fit_names{i}))
    disp('Mean AIC')
    disp(mean(aic(:,i)))
    disp('Mean BIC')
    disp(mean(bic(:,i)))
    
    subplot(2,5,i)
    imagesc(reshape(aic(:,i), 48, 48)')
    title(sprintf('%s AIC', Y_fit_names{i}))
    colorbar
    axis square; set(gca, 'XLim', [0 48], 'YLim', [0 48], 'FontSize', 8);
    
    subplot(2,5,5+i)
    imagesc(reshape(bic(:,i), 48, 48)')
    title(sprintf('%s BIC', Y_fit_names{i}))
    colorbar
    axis square; set(gca, 'XLim', [0 48], 'YLim', [0 48], 'FontSize', 8);
end
% 
set(gcf, 'PaperPosition', [0 0 16 9], 'PaperSize', [16 9])
% print(sprintf('./fig/ic/aic_bic_map-%g.png', dir_idx), '-dpng')
% 
% 
%% Show AIC/BIC grid
aic_grid = reshape(aic, 48, 48, 5);
bic_grid = reshape(bic, 48, 48, 5);

[~, aic_min] = min(aic_grid, [], 5);
[~, bic_min] = min(bic_grid, [], 5);

figure('PaperPosition', [0 0 10 10], 'PaperSize', [10 10]);
% subplot(2,1,1)
imagesc(permute(aic_min,[3 1 2]))
% title('AIC')
axis square; axis off;
set(gca, 'XLim', [0 48], 'YLim', [0 48], 'FontSize', 8);

% 
% figure('PaperPosition', [0 0 10 10], 'PaperSize', [10 10]);
% % subplot(2,1,2)
% imagesc(bic_min')
% % title('BIC')
% axis square; axis off;
% set(gca, 'XLim', [0 48], 'YLim', [0 48], 'FontSize', 8);
% print(sprintf('./fig/ic/BIC_model_select_map-%g.png', dir_idx), '-dpng')
% 
% 
% set(gcf, 'PaperPosition', [0 0 9 16], 'PaperSize', [9 16])% print(sprintf('./fig/ic/model_select_map-%g.png', dir_idx), '-dpng')



% 
% % Blue: linear model
% % Teal: quadratic model
% % Yellow: cubic model
% % APPLY MASK?