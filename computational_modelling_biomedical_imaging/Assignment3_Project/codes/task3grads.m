%% task 3 - inhalation/exhalation
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
% Extract the first 100 gradients - the array is shifted by one e.g. x=2 is classed
% as an inhalation if the signal value increases from x=1 to x=2; 
grads = gradient(SurrogateSignal(1,2:n_train+1));
% grads = gradient(x);
% registrations are separated into those from end-inhalation/end=exhalation
grads_inhale = grads(grads>0);
grads_exhale = grads(grads<0);
inhalations = x(grads>0); % troughs of signal correspond to inhalation 
exhalations = x(grads<0); % peaks correspond to exhalation

%% Extract SI displacement for CP 30, 20
si_disp = zeros(100,1);
% Extract the CP displacements
for t=1:n_train
    si_disp(t, :) = reshape(CPs{t}.img(30,20,1,1,2), 1, []);
end
% extract SI displacement corresponding to inhalations/exhalations 
si_disp_inhale = si_disp(grads>0);
si_disp_exhale = si_disp(grads<0);

%% Fit and plot the SI displacement separately for inhalations and exhalations
n_train_inhale = length(inhalations);
n_train_exhale = length(exhalations);

%% inhalations

% linear
X_inhale = [inhalations, ones(n_train_inhale, 1)];
y_inhale = si_disp_inhale;
w_inhale = X_inhale\y_inhale;
y_fit_inhale = X_inhale*w_inhale;

% (b) more sophisticated model (1D)
X_inhale_b = [grads_inhale', inhalations, ones(size(inhalations))]; 
w_inhale_b = X_inhale_b\y_inhale;
y_fit_inhale_b = X_inhale_b*w_inhale_b;

% quadratic 
X_quad_inhale = [inhalations.^2, inhalations, ones(n_train_inhale, 1)]; 
w_quad_inhale = X_quad_inhale\y_inhale;
y_quad_fit_inhale = X_quad_inhale*w_quad_inhale;

% (c) more sophisticated model (1D)
X_inhale_c = [grads_inhale', inhalations.^2, inhalations, ones(size(inhalations))]; 
w_inhale_c = X_inhale_c\y_inhale;
y_fit_inhale_c = X_inhale_c*w_inhale_c;

% cubic
X_cubic_inhale = [inhalations.^3, inhalations.^2, inhalations,... 
                                  ones(n_train_inhale, 1)]; 
w_cubic_inhale = X_cubic_inhale\y_inhale;
y_cubic_fit_inhale = X_cubic_inhale*w_cubic_inhale;

% exhalations
X_exhale = [exhalations, ones(n_train_exhale, 1)];
y_exhale = si_disp_exhale;
w_exhale = X_exhale\y_exhale;
y_fit_exhale = X_exhale*w_exhale;

% (b) more sophisticated model (2D)
X_exhale_b = [grads_exhale', exhalations, ones(size(exhalations))]; 
w_exhale_b = X_exhale_b\y_exhale;
y_fit_exhale_b = X_exhale_b*w_exhale_b;

% quadratic
X_quad_exhale = [exhalations.^2, exhalations, ones(n_train_exhale, 1)]; 
w_quad_exhale = X_quad_exhale\y_exhale;
y_quad_fit_exhale = X_quad_exhale*w_quad_exhale;

% (c) more sophisticated model (1D)
X_exhale_c = [grads_exhale', exhalations.^2, exhalations, ones(size(exhalations))]; 
w_exhale_c = X_exhale_c\y_exhale;
y_fit_exhale_c = X_exhale_c*w_exhale_c;

% cubic
X_cubic_exhale = [exhalations.^3, exhalations.^2, exhalations,... 
                                  ones(n_train_exhale, 1)]; 
w_cubic_exhale = X_cubic_exhale\y_exhale;
y_cubic_fit_exhale = X_cubic_exhale*w_cubic_exhale;

% plots of points/fit
figure();
plot(inhalations, y_fit_inhale,'k')
hold on
scatter(inhalations, y_inhale, 40, 'k', 'Marker', 'o')
plot(exhalations, y_fit_exhale, 'magenta')
scatter(exhalations, y_exhale, 45, 'magenta', 'Marker', 'x')
h = zeros(2, 1);
h(1) = plot(NaN,NaN,'-ok');
h(2) = plot(NaN,NaN,'-xm');
legend(h, 'end-inhalation','end-exhalation');
hold off

figure();
plot(inhalations, y_fit_inhale_b,'k')
hold on
scatter(inhalations, y_inhale, 40, 'k', 'Marker', 'o')
plot(exhalations, y_fit_exhale_b, 'magenta')
scatter(exhalations, y_exhale, 45, 'magenta', 'Marker', 'x')
h = zeros(2, 1);
h(1) = plot(NaN,NaN,'-ok');
h(2) = plot(NaN,NaN,'-xm');
legend(h, 'end-inhalation','end-exhalation');
hold off

figure();
plot(inhalations, y_fit_inhale_c,'k')
hold on
scatter(inhalations, y_inhale, 40, 'k', 'Marker', 'o')
plot(exhalations, y_fit_exhale_c, 'magenta')
scatter(exhalations, y_exhale, 45, 'magenta', 'Marker', 'x')
h = zeros(2, 1);
h(1) = plot(NaN,NaN,'-ok');
h(2) = plot(NaN,NaN,'-xm');
legend(h, 'end-inhalation','end-exhalation');
hold off

figure();
plot(inhalations, y_cubic_fit_inhale, 'k')
hold on
scatter(inhalations, y_inhale, 40, 'k', 'Marker', 'o')
plot(exhalations, y_cubic_fit_exhale, 'magenta')
scatter(exhalations, y_exhale, 45, 'magenta', 'Marker', 'x')
h = zeros(2, 1);
h(1) = plot(NaN,NaN,'-ok');
h(2) = plot(NaN,NaN,'-xm');
legend(h, 'end-inhalation','end-exhalation');
hold off

figure();
plot(inhalations, y_cubic_fit_inhale,'k')
hold on
scatter(inhalations, y_inhale, 40, 'k', 'Marker', 'o')
plot(exhalations, y_cubic_fit_exhale, 'magenta')
scatter(exhalations, y_exhale, 45, 'magenta', 'Marker', 'x')
h = zeros(2, 1);
h(1) = plot(NaN,NaN,'-ok');
h(2) = plot(NaN,NaN,'-xm');
legend(h, 'end-inhalation','end-exhalation');
hold off

% %% Extract SI displacement for CP 30, 20
% si_disp = zeros(100,1);
% for i=1:n_train
%     si_disp(i) = CPs{i}.img(30,20,1,1,2);
% end
% Y_inhale = si_disp_inhale;  % n_train x n_cp
% % Design matrix 
% X_inhale = [inhalations, ones(n_train_inhale, 1)];  % n_train x 2
% % W should be 2 x n_cp
% W_inhale = X_inhale\Y_inhale;
%  % Predicted surrogate signal
% Y_fit_inhale = X_inhale*W_inhale;
% 
% % exhalations
% si_disp_exhale = zeros(n_train_exhale, n_cp);
% % Extract the CP displacements (inhalation)
% for t=1:n_train_exhale
%     si_disp_exhale(t, :) = reshape(CPs{t}.img(:,:,1,1,2), 1, []);
% end
% Y_exhale = si_disp_exhale;  % n_train x n_cp
% % Design matrix 
% X_exhale = [exhalations, ones(n_train_exhale, 1)];  % n_train x 2
%  % W should be 2 x n_cp
% W_exhale = X_exhale\Y_exhale;
%  % Predicted surrogate signal
% Y_fit_exhale = X_exhale*W_exhale;
% 
% %% Check fits for CP 30, 20
% ind = sub2ind(size(CPs{1}.img), 30, 20);
% % inhalation
% disp('Inhalation: Coefficients of first fit on CP 30, 20:')
% disp(w_inhale)
% disp('Inhalation: Coefficients with vectorised operation on CP 30, 20:')
% disp(W_inhale(:, ind))
% 
% % exhalation
% disp('Exhalation: Coefficients of first fit on CP 30, 20: ')
% disp(w_exhale)
% disp('Exhalation: Coefficients with vectorised operation on CP 30, 20: ')
% disp(W_exhale(:, ind))
% 
% %% Visualize fits for some random CP's
% 
% % inhalation
% samp_size = 10;
% figure();
% for i=randsample(n_cp, samp_size)
%     plot(inhalations, Y_inhale(:,i), 'LineStyle', ':', 'Marker', 'x');
%     hold on
%     plot(inhalations, Y_fit_inhale(:,i))
% end
% hold off
% 
% % exhalation
% samp_size = 10;
% figure();
% for i=randsample(n_cp, samp_size)
%     plot(exhalations, Y_exhale(:,i), 'LineStyle', ':', 'Marker', 'x');
%     hold on
%     plot(exhalations, Y_fit_exhale(:,i))
% end
% hold off