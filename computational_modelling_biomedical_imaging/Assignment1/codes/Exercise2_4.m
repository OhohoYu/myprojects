%%%%%%%%%%%%%%%%%%%%%%%%
% EXERCISE2
% PARAMETRIC BOOTSTRAP
%%%%%%%%%%%%%%%%%%%%%%%%

% clear history
clear all

% load HARDI data
load('data');
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);

% load gradient directions
qhat = load('bvecs');

% construct b-value array
bvals = 1000*sum(qhat.*qhat);

% total number of bootstrap iterations
bootstrap_iterations = 100;
iterations_for_glob_min = 2;

% define different fitting options
h=optimset( 'MaxFunEvals',  20000, ...
            'Algorithm' , 'levenberg-marquardt', ...  
            'TolX' ,1e-10, 'TolFun' ,1e-10, ...
            'Display', 'off', 'LargeScale', 'off');
        
dwis = dwis(:,70:90,70:120,:);        

xvoxels = 50;
yvoxels = 20;

twosigmamattthetalow = zeros(xvoxels,yvoxels);
twosigmamattthetahigh = zeros(xvoxels, yvoxels);
ninefivematthetalow =  zeros(xvoxels, yvoxels);
ninefivematthetahigh = zeros(xvoxels, yvoxels);
twosigmamatphilow = zeros(xvoxels, yvoxels);
twosigmamatphihigh = zeros(xvoxels, yvoxels);
ninefivematphilow = zeros(xvoxels, yvoxels);
ninefivematphihigh =  zeros(xvoxels, yvoxels);
theta_mat = zeros(xvoxels,yvoxels);
phi_mat = zeros(xvoxels,yvoxels);
f_mat = zeros(xvoxels, yvoxels);

%%
% loop over different voxels
for y = 1:xvoxels
    for z=1:yvoxels
        % extract set of 108 measurements from one single voxel, 
        Avox = dwis(:,z,y,72);
        % arbitrary start point
        startx = [3.5e+0   3e-03   2.5e-01 0 0];
        min_resnorm = 80000000000;
        min_params = zeros(1,5);
        % create normally distributed numbers for each param
        random1 = randn(1,iterations_for_glob_min);
        random2 = randn(1,iterations_for_glob_min);
        random3 = randn(1,iterations_for_glob_min);
        random4 = randn(1,iterations_for_glob_min)*2*pi;
        random5 = randn(1,iterations_for_glob_min)*2*pi;
        % find globl minimum for each voxel
        for k=1:iterations_for_glob_min            
            new_startx(1) = startx(1) + (startx(1)*random1(k)); 
            new_startx(2) = startx(2) +  (startx(2)*random2(k));
            new_startx(3) = startx(3) +  (startx(3)*random3(k));
            new_startx(4) = startx(4)+random4(k);
            new_startx(5) = startx(5)+random5(k);
            try
            % Now run the fitting
            [parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSDPart1_2',...
                                                             new_startx, h,Avox,...
                                                             bvals,qhat);       
            catch
            new_startx = startx;     
            end                                             
            if RESNORM < min_resnorm 
                min_resnorm = RESNORM;
                min_params = parameter_hat;
            end
        end
        min_params(1) = min_params(1)^2; 
        min_params(2) = min_params(2)^2;
        min_params(3) = (1/(1+exp(-min_params(3))));
        theta_mat(y,z)=min_params(4);
        phi_mat(y,z)=min_params(5);
        f_mat(y,z)=min_params(3);
        % apply ball and stick function  (unconstrained)
        S = BallStick(min_params, bvals, qhat)';
        % compute initial RESNORM
        RESNORM_init = sum((Avox - S).^2);
        constant = sqrt(RESNORM_init/103);
        min_params(1) = min_params(1)/1000;
        startxstartx = min_params;
        thetalist = zeros(1,bootstrap_iterations);
        philist = zeros(1,bootstrap_iterations);
        for l=1:bootstrap_iterations 
            S_new = S + constant.*randn(size(S,1), 1);
            try
            [parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSDPart1_2',...
                                                            startxstartx, h, ...
                                                            S_new,bvals,qhat); 
            catch
            startxstartx = min_params;
            end
            thetalist(l) = parameter_hat(4);
            philist(l) = parameter_hat(5);
        end
        % compute means and stdev
        mean_theta = mean(thetalist);
        stdev_theta = std(philist);
        mean_phi = mean(thetalist);
        stdev_phi = std(philist);
        twosigrangetheta = [mean_theta - 2*stdev_theta, mean_theta + 2*stdev_theta];
        ninefiverangetheta = [prctile(thetalist,2.5), prctile(thetalist,97.5)];
        twosigrangephi = [mean_phi - 2*stdev_phi, mean_phi + 2*stdev_phi];
        ninefiverangephi = [prctile(philist,2.5), prctile(philist,97.5)];
        twosigmamattthetalow(y,z) = twosigrangetheta(1);
        twosigmamattthetahigh(y,z) = twosigrangetheta(2);
        ninefivematthetalow(y,z) =  ninefiverangetheta(1);
        ninefivematthetahigh(y,z) = ninefiverangetheta(2);
        twosigmamatphilow(y,z) = twosigrangephi(1);
        twosigmamatphihigh(y,z) = twosigrangephi(2);
        ninefivematphilow(y,z) = ninefiverangephi(1);
        ninefivematphihigh(y,z) =  ninefiverangephi(2);
    end
    y
end   

% FIBRE DIRECTION PLOTS
[xfib,yfib,zfib]=sph2cart(theta_mat, phi_mat, ones(xvoxels,yvoxels));
figure
quiver(xfib.*f_mat,yfib.*f_mat, 1.5)
[xfib2,yfib2,zfib2]=sph2cart(twosigmamattthetalow, twosigmamatphilow, ones(xvoxels,yvoxels));
hold on
quiver(xfib2.*f_mat,yfib2.*f_mat, 1.5)
[xfib3,yfib3,zfib3]=sph2cart(twosigmamattthetahigh, twosigmamatphihigh, ones(xvoxels,yvoxels));
quiver(xfib3.*f_mat,yfib3.*f_mat, 1.5)
hold off

figure
quiver(xfib.*f_mat,yfib.*f_mat, 1.5)
[xfib4,yfib4,zfib4]=sph2cart(ninefivematthetalow, ninefivematphilow, ones(xvoxels,yvoxels));
hold on
quiver(xfib4.*f_mat,yfib4.*f_mat, 1.5)
[xfib5,yfib5,zfib5]=sph2cart(ninefivematthetahigh, ninefivematphihigh, ones(xvoxels,yvoxels));
quiver(xfib5.*f_mat,yfib5.*f_mat, 1.5)
hold off


