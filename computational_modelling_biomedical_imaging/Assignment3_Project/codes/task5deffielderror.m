%% RUN TASK 5 (Task_5.m) BEFORE
%% DEFORMATION FIELD ERROR:
% Load mask to compute only errors in the ROI
mask = load_untouch_nii('mask.nii');

% load source images nii
imgs = dir('data/images/IM*.nii');
srcIMGs = cell(n_test, 1);
for i=1:n_test
    srcIMGs{i} = load_untouch_nii(imgs(n_train + i).name);
end

% load true CPs
trueCPs = cell(n_test,1);
for i=1:n_test
    trueCPs{i} = load_untouch_nii(regs(n_train + i).name);
end

% load target nii
target_nii = load_untouch_nii('IM_0006.nii');

n_pix = numel(target_nii.img);
predDist = zeros(n_test, n_pix);
for t=1:n_test
    % Compute true deformation field
    [~, true_def_field_nii] = deformNiiWithCPG(trueCPs{t}, srcIMGs{t}, target_nii);
    
    % Compute predicted deformation field
    [~, pred_def_field_nii] = deformNiiWithCPG(predCPs{t}, srcIMGs{t}, target_nii);
    
    % Reshape mask and convert to double
    maskCoords = double(repmat(reshape(mask.img, [], 1), [1 2]));
    
    % Element wise multiplication of deformation field with mask to set to
    % 0 all values outside mask
    trueDefFieldCoords = reshape(true_def_field_nii.img, [], 2);
    % trueDefFieldCoords = reshape(true_def_field_nii.img, [], 2) .* maskCoords;
    % predDefFieldCoords = reshape(pred_def_field_nii.img, [], 2) .* maskCoords;
    
    % Compute L2 norm of vector at each pixel between prediction and true
    % deformation field
    
    
    % predDist(t, :) = sqrt(sum(abs(trueDefFieldCoords - predDefFieldCoords).^2,2));
    
end

% Plot heatmap of deformation field error over test images
figure;
set(gca,'DataAspectRatio',[1,1,1]);
cmax = max(predDist(:));
for t=1:n_test
    imagesc(reshape(trueDefFieldCoords(t, :), 480, 480)', [0 cmax]); colorbar;
    drawnow;
end

% Plot deformation Field error per image overlayed with surrogate signal to
% correlate error and respiratory phase
figure;
yyaxis left
plot(sum(predDist, 2));
xlabel('Images')
ylabel('Deformation Field Error');
yyaxis right
plot(xTest);
ylabel('Surrogate Signal - AP displacement');

% Plot histogram to sum up results over all test images
figure;
histogram(sum(predDist, 2));
xlabel('Deformation Field Error')
ylabel('Image counts');