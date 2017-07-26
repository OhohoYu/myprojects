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

% extract set of 108 measurements from one single voxel, 
% take one near the centre of the image
Avox = dwis(:,92,65,72);

% Define a starting point for the non-linear fit
startx = [3.5e+0   3e-03   2.5e-01 0 0];

% Define various options for the non-linear fitting % algorithm.
h=optimset( 'MaxFunEvals',  20000, ...
            'Algorithm' , 'levenberg-marquardt',   ...  
            'TolX' ,1e-10, 'TolFun' ,1e-10, 'Display', 'off',...
            'LargeScale', 'off');
% Now run the fitting
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSD',startx,h, ...
                                                 Avox,bvals,qhat);

%%%%%%%%%%%%%%%%%%%
% Exercise 1.1.1                                           
%%%%%%%%%%%%%%%%%%%

% average and standard deviation for measured signal
[RESNORM_check, S_pred] = BallStickSSD(parameter_hat, Avox, bvals, qhat);
avg_S_pred = mean(S_pred);
stdev_S_pred = std(S_pred);

% average and standard deviation for predicted signal
avg_S_meas = mean(Avox);
stdev_S_meas = std(Avox);

% plot actual measurements and parametrised model predictions
figure; 
x1 = 1:108;
y1 = S_pred;
y2 = Avox;
plot(x1,y1, 'ob')
hold on
plot(x1,y2, 'xr')
xlabel('measurement')
ylabel('S')
ylim([1000 5000])
legend('predicted S', 'measured S')
hold off

%%%%%%%%%%%%%%%%%
% Exercise 1.1.2
%%%%%%%%%%%%%%%%%
% We make use of modified Ball and Stick function BallStickSSDPart1_2.m


t = cputime;

% inverse transformation on starting point
newstartx(1) = sqrt(startx(1));
newstartx(2) = sqrt(startx(2));
newstartx(3) = -log((1/startx(3))-1);
newstartx(4) = startx(4);
newstartx(5) = startx(5);


% Now rerun the fitting
[parameter_hat2,RESNORM2,EXITFLAG2,OUTPUT2]=fminunc('BallStickSSDPart1_2',newstartx,h, ...
                                                     Avox,bvals,qhat);
                                               
                                                 
 e = cputime-t
                                                 
% average and standard deviation for predicted signal
[RESNORM_check2, S_pred2] = BallStickSSDPart1_2(parameter_hat2, Avox, bvals, qhat);
avg_S_pred2 = mean(S_pred2);
stdev_S_pred2 = std(S_pred2);

% average and standard deviation for measured signal
avg_S_meas2 = mean(Avox);
stdev_S_meas2 = std(Avox);

% reapply transformation to recover parameters
parameter_hat2(1) = parameter_hat2(1)^2; 
parameter_hat2(2) = parameter_hat2(2)^2;
parameter_hat2(3) = (1/(1+exp(-parameter_hat2(3))));

% plot actual measurements and parametrised model predictions
figure; 
x1 = 1:108;
y1 = S_pred2;
y2 = Avox;
hold on
plot(x1,y1, 'ob')
plot(x1,y2, 'xr')
xlabel('measurement')
ylabel('S')
ylim([1000 5000])
legend('predicted S', 'measured S')
hold off