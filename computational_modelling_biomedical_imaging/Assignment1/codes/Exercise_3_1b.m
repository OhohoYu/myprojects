% clear history
clear all

% Load the diffusion signal %
fid = fopen( 'isbi2015_data_normalised.txt','r' ,'b'); 
fgetl(fid);  % Read in the header
D = fscanf(fid,'%f',[6,inf])'; % Read in the data 
fclose(fid);

% Select the first of the 6 voxels
meas = D(:,1);

% Load the protocol %
fid = fopen('isbi2015_protocol.txt','r','b'); 
fgetl(fid);
A = fscanf(fid,'%f',[7,inf]);
fclose(fid);

% Create the protocol
grad_dirs = A(1:3,:); 
G = A(4,:);
delta = A(5,:); 
smalldel = A(6,:); 
TE = A(7,:);

GAMMA =  2.675987E8;
bvals = ((GAMMA*smalldel.*G).^2).*(delta - smalldel/3);

%% EQUIVALENT OF PART 1.1.3
%%%%%%%%
% set random seed for reproducibility
rng(4);
% Define a starting point for the non-linear fit
startx = [3.5e+0   3e-09   2.5e-01 0 0];


% iterations to find minimum for given settings
iters_for_min = 200;

% create normally distributed numbers for each param
random1 = randn(1,iters_for_min);
random2 = randn(1,iters_for_min);
random3 = randn(1,iters_for_min);
random4 = randn(1,iters_for_min)*2*pi;
random5 = randn(1,iters_for_min)*2*pi;

% this array keeps track of all RESNORM
RESNORM_array = zeros(1,iters_for_min);

% initialise minimum RESNORM
min_RESNORM = 30000000000000;

% find minimum first 
for i=1:iters_for_min
    startx(1) = startx(1) + (startx(1)*random1(i)); 
    startx(2) = startx(2) +  (startx(2)*random2(i));
    startx(3) = startx(3) +  (startx(3)*random3(i));
    startx(4) = startx(4)+random4(i);
    startx(5) = startx(5)+random5(i);
    % Define various options for the non-linear fitting % algorithm.
    h=optimset( 'MaxFunEvals',  20000, ...
                'Algorithm' , 'levenberg-marquardt',   ...  
                'TolX' ,1e-10, 'TolFun' ,1e-10, ...
                'Display', 'off', 'LargeScale', 'off');
    % Now run the fitting
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSDPart1_2',...
                                                     startx, h,meas,bvals,grad_dirs);
                                                 
    RESNORM_array(i)=RESNORM;
    if RESNORM < min_RESNORM
        % stores minimum 
        min_RESNORM = RESNORM; 
    end
    i
end

sum(RESNORM_array < min_RESNORM + 0.00001)