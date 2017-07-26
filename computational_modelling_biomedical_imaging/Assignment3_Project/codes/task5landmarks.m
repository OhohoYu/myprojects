%% Landmark error

% Load landmark file:
load('landmarks.mat');

Images = dir(['data' filesep 'images' filesep 'IM*.nii']);
regs = dir(['data' filesep 'reg_results' filesep 'Reg*.nii']);

% get test images
niis = cell(200,1);
for i = 1: 200
    niis{i} = load_untouch_nii(Images(i + 100).name);
end

% Load reference image. 
RefNii = load_untouch_nii(Images(6).name);

% Just on test images:
test = 200;
n_train = 100;

%% Get predictions for si and ap from all models:

si_disp_mat = zeros(n_train, n_cp);
ap_disp_mat = zeros(n_train, n_cp);

% Extract the first 100 surrogate signal measurements
x = SurrogateSignal(1, 1:n_train)';

% Extract the CP displacements
for t=1:n_train
    si_disp_mat(t, :) = reshape(CPs{t}.img(:,:,1,1,2), 1, []);
end

% Extract the CP displacements
for t=1:n_train
    ap_disp_mat(t, :) = reshape(CPs{t}.img(:,:,1,1,2), 1, []);
end

Y_si = si_disp_mat;  % n_train x n_cp
Y_ap = si_disp_mat;

% Fit first-order (linear) model (again)
X_linear = [x, ones(size(x))];
W_linear_si = X_linear\Y_si;
W_linear_ap = X_linear\Y_ap;

% Fit second-order (quadratic) model
X_quad = [x.^2, x, ones(n_train, 1)]; 
W_quad_si = X_quad\Y_si;
W_quad_ap = X_quad\Y_ap;

% Fit third-order (cubic) model
X_cubic = [x.^3, x.^2, x, ones(size(x))];
W_cubic_si = X_cubic\Y_si;
W_cubic_ap = X_cubic\Y_ap;


%% Predict new regs from all models
n_test = 200;

% Extract the last 200 surrogate signal measurements
x = SurrogateSignal(1, n_train+1:length(SurrogateSignal))';

% Linear prediction:
X_linear = [x, ones(size(x))];
fit_linear_si = X_linear*W_linear_si;
fit_linear_ap = X_linear*W_linear_ap;

% Quadratic prediction:
X_quad = [x.^2, x, ones(size(x))]; 
fit_quad_si = X_quad*W_quad_si;
fit_quad_ap = X_quad*W_quad_si;

% Cubic prediction
X_cubic = [x.^3, x.^2, x, ones(size(x))];
fit_cubic_si = X_cubic*W_cubic_si;
fit_cubic_ap = X_cubic*W_cubic_ap;

%% Store all these prediction in nifity format (needed for function)

% load some nifity files from which we can can .img data
CPs_l = cell(200,1);
CPs_q = cell(200,1);
CPs_c = cell(200,1);

for i= 1:n_test
    CPs_l{i} = load_untouch_nii(regs(i + 100).name);
    CPs_q{i} = load_untouch_nii(regs(i + 100).name);
    CPs_c{i} = load_untouch_nii(regs(i + 100).name);
    % + 100 probably doesn't make a difference
end

% replace the nifity.img which predictions
for i = 1:test
    % linear
    lin_fit(:,:,1,1,1) = reshape(fit_linear_ap(i,:), 48, 48, 1, 1, 1);
    lin_fit(:,:,1,1,2) = reshape(fit_linear_si(i,:), 48, 48, 1, 1, 1);
    CPs_l{i}.img = lin_fit;
    
    % Quadratic
    quad_fit(:,:,1,1,1) = reshape(fit_quad_ap(i,:), 48, 48, 1, 1, 1);
    quad_fit(:,:,1,1,2) = reshape(fit_quad_si(i,:), 48, 48, 1, 1, 1);
    CPs_q{i}.img = quad_fit;
    
    % Cubic
    cubic_fit(:,:,1,1,1) = reshape(fit_cubic_ap(i,:), 48, 48, 1, 1, 1);
    cubic_fit(:,:,1,1,2) = reshape(fit_cubic_si(i,:), 48, 48, 1, 1, 1);
    CPs_c{i}.img = cubic_fit;
end


%% Apply predicted transformation and retreive predicted landmark

% Use the given function to find the predicted landmark

pred_lk_linear = zeros(3,2,300);
pred_lk_quad = zeros(3,2,300);
pred_lk_cubic = zeros(3,2,300);
for i = 1:200
    pred_lk_linear(:,:,i) = transPointsWithCPG(CPs_l{i}, landmark_pos(:,:,i+100), false);
    pred_lk_quad(:,:,i) = transPointsWithCPG(CPs_q{i}, landmark_pos(:,:,i+100), false);
    pred_lk_cubic(:,:,i) = transPointsWithCPG(CPs_c{i}, landmark_pos(:,:,i+100), false);
end
%% Find Errors

% Not sure of images below are correct. Will have carry on once I ask james

%% Display movement in landmarks
figure()
% Display reference image:
%imshow(RefNii.img',[])
%hold on 
%plot(landmark_pos(:,1,8), landmark_pos(:,2,8),'r*')


for i = 1 : length(Images)
    % load image
    cur_image = load_untouch_nii(Images(i).name);
    plot(2,2);
    subplot(2,2,1)
    imshow(cur_image.img', [])
    hold on
    plot(landmark_pos(:,1,i), landmark_pos(:,2,i),'r*', 'markersize', 1)
    hold off
    
    subplot(2,2,2)
    imshow(cur_image.img', [])
    hold on
    plot(pred_lk_linear(:,1,i)', pred_lk_linear(:,2,i)','b*', 'markersize', 1)
    hold off
    
    subplot(2,2,3)
    imshow(cur_image.img', [])
    hold on
    plot(pred_lk_quad(:,1,i)', pred_lk_quad(:,2,i)','b*', 'markersize', 1)
    hold off
    
    subplot(2,2,4)
    imshow(cur_image.img', [])
    hold on
    plot(pred_lk_cubic(:,1,i)', pred_lk_cubic(:,2,i)','b*', 'markersize', 1)
    hold off
   
    
    pause(0.1)
    
end