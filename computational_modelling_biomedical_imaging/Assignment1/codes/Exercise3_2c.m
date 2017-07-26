% clear history
clear all
rng(25)

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

% Define a starting point for the non-linear fit
startx = [3.5e+0   3e-09   2.5e-01 0 0];

random1 = randn();
random2 = randn();
random3 = randn();
random4 = randn()*2*pi;
random5 = randn()*2*pi;

startx(1) = startx(1) + (startx(1)*random1); 
startx(2) = startx(2) +  (startx(2)*random2);
startx(3) = startx(3) +  (startx(3)*random3);
startx(4) = startx(4)+random4;
startx(5) = startx(5)+random5;

random1 = randn();
random2 = randn();
random3 = randn();
random4 = randn()*2*pi;
random5 = randn()*2*pi;

startx(1) = startx(1) + (startx(1)*random1); 
startx(2) = startx(2) +  (startx(2)*random2);
startx(3) = startx(3) +  (startx(3)*random3);
startx(4) = startx(4)+random4;
startx(5) = startx(5)+random5;



% Define various options for the non-linear fitting % algorithm.
h=optimset( 'MaxFunEvals',  20000, ...
            'Algorithm' , 'levenberg-marquardt',   ...  
            'TolX' ,1e-10, 'TolFun' ,1e-10, 'Display', 'off',...
            'LargeScale', 'off');

%%%%%%%%%%%%%%%%%
% Exercise 1.1.2
%%%%%%%%%%%%%%%%%
% We make use of modified Ball and Stick function BallStickSSDPart1_2.m

% inverse transformation on starting point
newstartx(1) = sqrt(startx(1));
newstartx(2) = sqrt(startx(2));
newstartx(3) = -log((1/startx(3))-1);
newstartx(4) = startx(4);
newstartx(5) = startx(5);


% Now rerun the fitting
[parameter_hat2,RESNORM2,EXITFLAG2,OUTPUT2]=fminunc('DiffusionTensorSSD',newstartx,h, ...
                                                     meas,bvals,grad_dirs);
                                                                                             

% average and standard deviation for predicted signal
[RESNORM_check2, S_pred2] = DiffusionTensorSSD(parameter_hat2, meas, bvals, grad_dirs);
avg_S_pred2 = mean(S_pred2);
stdev_S_pred2 = std(S_pred2);

% average and standard deviation for measured signal
avg_S_meas2 = mean(meas);
stdev_S_meas2 = std(meas);

% reapply transformation to recover parameters
parameter_hat2(1) = parameter_hat2(1)^2; 
parameter_hat2(2) = parameter_hat2(2)^2;
parameter_hat2(3) = (1/(1+exp(-parameter_hat2(3))));

% plot actual measurements and parametrised model predictions
figure; 
x1 = 1:3612;
y1 = S_pred2;
y2 = meas;
hold on
plot(x1,y1, 'ob')
plot(x1,y2, 'xr')
xlabel('measurement')
ylabel('S')
legend('predicted S', 'measured S')
hold off