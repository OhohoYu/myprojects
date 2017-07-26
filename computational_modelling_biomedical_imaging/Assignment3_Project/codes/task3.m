%% TASK 3: FIT A LINEAR MODEL
close all

%% Add functions and data paths
% addpath(genpath('data'))
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

%% Extract SI displacement for CP 30, 20
si_disp = zeros(100,1);
for i=1:n_train
    si_disp(i) = CPs{i}.img(30,20,1,1,2);
end
    
%% Fit and plot the SI displacement
X = [x, ones(n_train, 1)];
y = si_disp;
w = X\y;
y_fit = X*w;

Xq = [x.^2, x, ones(n_train, 1)];
yq= si_disp;
wq = Xq\yq;
y_quad_fit = Xq*wq;

Xc = [x.^3, x.^2, x, ones(n_train, 1)];
yc= si_disp;
wc = Xc\yc;
y_cub_fit = Xc*wc;



figure();

plot(x, y, 'LineStyle', ':', 'Marker', 'o','Color', 'r', 'Markersize', 6, 'Markerfacecolor','r')
hold on
plot(x, y_fit, 'LineWidth', 2, 'Color', 'b')

%% Fit all of the CP displacements
n_cp = numel(CPs{1}.img(:,:,1,1,2));
si_disp_mat = zeros(n_train, n_cp);

% Extract the CP displacements
for t=1:n_train
    si_disp_mat(t, :) = reshape(CPs{t}.img(:,:,1,1,2), 1, []);
end

Y = si_disp_mat;  % n_train x n_cp

% Design matrix X is same as before
X = [x, ones(n_train, 1)];  % n_train x 2

% W should be 2 x n_cp
W = X\Y;

% Predicted surrogate signal
Y_fit = X*W;

%% Check fits for CP 30, 20
ind = sub2ind(size(CPs{1}.img), 30, 20);
disp('Coefficients of first fit on CP 30, 20:')
disp(w)
disp('Coefficients with vectorised operation on CP 30, 20:')
disp(W(:, ind))

%% Visualize fits for some random CP's
samp_size = 10;
figure();
for i=randsample(n_cp, samp_size)
    plot(x, Y(:,i), 'LineStyle', ':', 'Marker', 'x');
    hold on
    plot(x, Y_fit(:,i))
end
hold off

%% Fit second-order (quadratic) model
X_quad = [x.^2, x, ones(n_train, 1)]; 
W_quad = X_quad\y;
% Predicted surrogate signal
Y_quad_fit = X_quad*W_quad;

%% Fit third-order (cubic) model
X_cubic = [x.^3, x.^2, x, ones(size(x))];
W_cubic = X_cubic\y;
Y_cubic_fit = X_cubic * W_cubic;

% figure();
% set(gcf,'color','w'); % set figure background to white
% 
% plot(x, y, 'LineStyle', ':', 'LineWidth',1.5,'Marker', 'x', 'MarkerSize',10)
% set(gca,'FontSize',15)
% xlabel('s', 'FontSize',25)
% ylabel('x', 'FontSize',25)
% 
% hold on
% plot(x, Y_quad_fit, 'LineWidth', 2.5)
% hold off

figure();

plot(x, yc, 'LineStyle', ':', 'Marker', 'o','Color', 'r', 'Markersize', 6, 'Markerfacecolor','r')
hold on
plot(x, y_cub_fit, 'LineWidth', 2, 'Color', 'b')



%% Compute residual fitting errors
errs = (Y_fit - Y).^2;
rss = sum(errs, 1);
total_rss = sum(rss);
mean_rss = mean(rss);

disp('=== Linear Fit ===')
disp('Total residual sum of squares:')
disp(total_rss)
disp('Mean residual sum of squares:')
disp(mean_rss);