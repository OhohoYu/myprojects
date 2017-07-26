% clear history
clear all

load('data');
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);

% Middle slice of the first image volume, which has b=0
imshow(flipud(squeeze(dwis(1,:,:,72))'), []);
% Middle slice of the second image volume with b=1000
imshow(flipud(squeeze(dwis(2,:,:,72))'), []);

% load gradient directions
qhat = load('bvecs');

% construct b-value array
bvals = 1000*sum(qhat.*qhat);

% number of iterations to find global minimum for each voxel
iters_for_glob_min=5;

xvoxels = 174;
yvoxels = 145;

S0_mat = zeros(xvoxels,yvoxels);
d_mat = zeros(xvoxels,yvoxels);
f_mat = zeros(xvoxels,yvoxels);
resnorm_mat = zeros(xvoxels,yvoxels);
theta_mat = zeros(xvoxels,yvoxels);
phi_mat = zeros(xvoxels,yvoxels);

h=optimset( 'MaxFunEvals',  20000, ...
            'Algorithm' , 'levenberg-marquardt',   ...  
            'TolX' ,1e-10, 'TolFun' ,1e-10, ...
            'Display', 'off', 'LargeScale', 'off');

% loop over each voxel
for y=1:xvoxels
    for z=1:yvoxels
        % extract set of 108 measurements from one single voxel, 
        Avox = dwis(:,z,y,72);
        % Define a starting point for the non-linear fit
        startx = [3.5e+0   3e-03   2.5e-01 0 0];
        min_resnorm = 80000000000;
        min_params = zeros(1,5);
        % create normally distributed numbers for each param
        random1 = randn(1,iters_for_glob_min);
        random2 = randn(1,iters_for_glob_min);
        random3 = randn(1,iters_for_glob_min);
        random4 = randn(1,iters_for_glob_min)*2*pi;
        random5 = randn(1,iters_for_glob_min)*2*pi;
        for k=1:iters_for_glob_min    
            new_startx(1) = startx(1) + (startx(1)*random1(k)); 
            new_startx(2) = startx(2) +  (startx(2)*random2(k));
            new_startx(3) = startx(3) +  (startx(3)*random3(k));
            new_startx(4) = startx(4)+random4(k);
            new_startx(5) = startx(5)+random5(k);  
            % Define various options for the non-linear fitting % algorithm.                               
            try    
            
            % Now run the fitting
            [parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSDPart1_2',...
                                                             new_startx, h,Avox,bvals,qhat);       
            catch
            new_startx = startx;  
            end
            if RESNORM < min_resnorm 
                min_resnorm = RESNORM;
                min_params = parameter_hat;
            end   
        end 
        % reapply transformation to recover parameters
        min_params(1) = min_params(1)^2; 
        min_params(2) = min_params(2)^2;
        min_params(3) = (1/(1+exp(-min_params(3))));
        
        S0_mat(y,z) = min_params(1);
        d_mat(y,z) = min_params(2);
        f_mat(y,z) = min_params(3);
        resnorm_mat(y,z) = min_resnorm;
        theta_mat(y,z) = min_params(4); 
        phi_mat(y,z) = min_params(5);
        z
    end
    y
end   

% PARAMETER PLOTS
surf(S0_mat)
figure
surf(d_mat)
figure
surf(f_mat)
figure
surf(resnorm_mat)

% FIBRE DIRECTION PLOTS
[xfib,yfib,zfib]=sph2cart(theta_mat, phi_mat, ones(xvoxels,yvoxels));
figure
quiver(xfib.*f_mat,yfib.*f_mat, 1.5)
