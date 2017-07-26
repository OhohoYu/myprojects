%% RUN TASK 5 (Task_5.m) BEFORE
%% VISUAL EVALUATION:
%load target nii
target_nii = load_untouch_nii('IM_0006.nii');

for t=1:n_test
    % load source image
    source_nii = load_untouch_nii(strcat('IM_0', num2str(n_train + t, '%03d'), '.nii'));
    % Deform image with previously predicted displacements
    def_vol_nii = deformNiiWithCPG(predCPs{t}, source_nii, target_nii);
   
    % Display deformed image
    h = dispNiiSlice(def_vol_nii,'z',1);
    drawnow;
end