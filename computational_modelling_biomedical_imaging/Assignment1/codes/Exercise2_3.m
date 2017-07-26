%%%%%%%%%%%%%%%%%%%%%%%%
% EXERCISE2.3 -> Similar to 2.1 but using Laplace's method
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
%%
% extract set of 108 measurements from one single voxel, 
% take one near the centre of the image
Avox = dwis(:,73,87,72);

% total number of bootstrap iterations
tot_iterations = 20000;
min_resnorm = 1000000000000;

% Define a starting point for the non-linear fit
startx = [3.5e+0   3e-03   2.5e-01 0 0];

% to find minimum
iterations_for_min = 10;

h=optimset( 'MaxFunEvals',  20000, ...
            'Algorithm' , 'levenberg-marquardt', ...  
            'TolX' ,1e-10, 'TolFun' ,1e-10, ...
            'Display', 'off', 'LargeScale', 'off');

% create normally distributed numbers for each param
random1 = randn(1,iterations_for_min);
random2 = randn(1,iterations_for_min);
random3 = randn(1,iterations_for_min);
random4 = randn(1,iterations_for_min)*2*pi;
random5 = randn(1,iterations_for_min)*2*pi;
for k=1:iterations_for_min            
    new_startx(1) = startx(1) + (startx(1)*random1(k)); 
    new_startx(2) = startx(2) +  (startx(2)*random2(k));
    new_startx(3) = startx(3) +  (startx(3)*random3(k));
    new_startx(4) = startx(4)+random4(k);
    new_startx(5) = startx(5)+random5(k);           
    % Now run the fitting
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT, grad, hessian]=fminunc('BallStickSSDPart1_2',...
                                                                   new_startx, h,Avox,...
                                                                   bvals,qhat);       
    if RESNORM < min_resnorm 
        min_resnorm = RESNORM;
        min_params = parameter_hat;
    end       
end
% recovery
min_params(1) = min_params(1)^2; 
min_params(2) = min_params(2)^2;
min_params(3) = (1/(1+exp(-min_params(3))));

% parameters for question Q1.1.2
best_params = min_params;

% apply ball and stick function 
S = BallStick(best_params, bvals, qhat)';
% compute initial RESNORM
RESNORM_init = sum((Avox - S).^2);
constant = sqrt(RESNORM_init/103);


startx = best_params;
startx(1) = startx(1)/1000;
S0list = zeros(1,tot_iterations);
difflist = zeros(1,tot_iterations);
flist = zeros(1, tot_iterations);

for i=1:tot_iterations
    % Define various options for the non-linear fitting % algorithm.    
    % Now run the fitting                                          
    S_new = S + constant.*randn(size(S,1), 1);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT, grad, hessian]=fminunc('BallStickSSDPart1_2',...
                                                           startx, h, ...
                                                           S_new,bvals,qhat); 
                                                  
    % reapply transformation to recover parameters
    parameter_hat(1) = parameter_hat(1)^2; 
    parameter_hat(2) = parameter_hat(2)^2;
    parameter_hat(3) = (1/(1+exp(-parameter_hat(3))));
    S0list(i) = parameter_hat(1);
    difflist(i) = parameter_hat(2);
    flist(i) = parameter_hat(3);
    yo = diag(hessian);
    yo(1)
end

% % hist(S0list, 50)
% % title('S0')
% % figure
% % hist(difflist, 50)
% % title('diff')
% % figure
% % hist(flist, 50)
% % title('f')
% % 
% % % compute means and stdev
% % mean_S0 = mean(S0list);
% % stdev_S0 = std(S0list);
% % mean_diff = mean(difflist);
% % stdev_diff = std(difflist);
% % mean_f = mean(flist);
% % stdev_f = std(flist);
% % % cdf plots
% % figure        
% % cdfplot(S0list)
% % % test1 = ((randn(20000,1))*stdev_S0)+mean_S0;
% % % hold on 
% % % cdfplot(test1)
% % % legend('Empirical CDF', 'Standard Gaussian', 'Location', 'NW')
% % figure
% cdfplot(difflist)
% % test2 = ((randn(20000,1))*stdev_diff)+mean_diff;
% % hold on 
% % cdfplot(test2)
% % legend('Empirical CDF', 'Standard Gaussian', 'Location', 'NW')
% figure
% cdfplot(flist)
% % test3 = ((randn(20000,1))*stdev_f)+mean_f;
% % hold on 
% % cdfplot(test3)
% % legend('Empirical CDF', 'Standard Gaussian', 'Location', 'NW')
% 
% % kolgomorov-smirnov statistics
% h1=kstest(S0list);
% h2=kstest(difflist);
% h3=kstest(flist);
% 
% % compute ranges
% twosigrangeS0 = [mean_S0 - 2*stdev_S0, mean_S0 + 2*stdev_S0];
% ninefiverangeS0 = [prctile(S0list,2.5), prctile(S0list,97.5)];
% twosigrangediff = [mean_diff - 2*stdev_diff, mean_diff + 2*stdev_diff];
% ninefiverangediff = [prctile(difflist,2.5), prctile(difflist,97.5)];
% twosigrangef = [mean_f- 2*stdev_f, mean_f + 2*stdev_f];
% ninefiverangef = [prctile(flist,2.5), prctile(flist,97.5)];