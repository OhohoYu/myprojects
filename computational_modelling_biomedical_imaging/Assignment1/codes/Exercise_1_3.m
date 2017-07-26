% clear history
clear all

% set random seed for reproducibility
rng(3)

% load HARDI data
load('data');
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);

% load gradient directions
qhat = load('bvecs');

% construct b-value array
bvals = 1000*sum(qhat.*qhat);

% extract set of 108 measurements from one single voxel, 
% take one near the centre of the image
Avox = dwis(:,92,65,72);

% Define a starting point for the non-linear fit
startx = [3.5e+0   3e-03   2.5e-01 0 0];

% iterations to find minimum for given settings
iters_for_min = 5000;

% create normally distributed numbers for each param
random1 = randn(1,iters_for_min);
random2 = randn(1,iters_for_min);
random3 = randn(1,iters_for_min);
random4 = randn(1,iters_for_min)*2*pi;
random5 = randn(1,iters_for_min)*2*pi;

% this array keeps track of all RESNORM
RESNORM_array = zeros(1,iters_for_min);

% initialise minimum RESNORM
min_RESNORM = 30000000000;

% find minimum first 
for i=1:iters_for_min
    new_startx(1) = startx(1) + (startx(1)*random1(i)); 
    new_startx(2) = startx(2) +  (startx(2)*random2(i));
    new_startx(3) = startx(3) +  (startx(3)*random3(i));
    new_startx(4) = startx(4)+random4(i);
    new_startx(5) = startx(5)+random5(i);
    % Define various options for the non-linear fitting % algorithm.
    h=optimset( 'MaxFunEvals',  20000, ...
                'Algorithm' , 'levenberg-marquardt',   ...  
                'TolX' ,1e-10, 'TolFun' ,1e-10, ...
                'Display', 'off', 'LargeScale', 'off');
    % Now run the fitting
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSDPart1_2',...
                                                     new_startx, h,Avox,bvals,qhat);
    RESNORM_array(i)=RESNORM;
    if RESNORM < min_RESNORM
        % stores minimum 
        min_RESNORM = RESNORM; 
    end
end

sum(RESNORM_array < min_RESNORM + 1)

