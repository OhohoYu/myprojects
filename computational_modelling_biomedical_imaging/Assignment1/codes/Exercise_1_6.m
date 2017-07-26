%%%%%%%%%%%%%%%%%
% Exercise 1.1.6
%%%%%%%%%%%%%%%%%
% clear history
clear all

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

% Define various options for the non-linear fitting % algorithm.
h=optimset( 'MaxFunEvals',  20000, ...
            'Algorithm' , 'active-set',   ...  
            'TolX' ,1e-10, 'TolFun' ,1e-10, 'Display', 'off',...
            'LargeScale', 'off');
% Now run the fitting
lb = [0; 0; 0; -inf; -inf];
ub = [inf; inf; 1; inf; inf];
t = cputime;
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fmincon('BallStickSSD',startx,[],[],[],[],lb, ub, [], h, Avox,bvals,qhat);
% cpu time
e = cputime - t
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
hold on
plot(x1,y1, 'ob')
plot(x1,y2, 'xr')
xlabel('measurement')
ylabel('S')
ylim([1000 5000])
legend('predicted S', 'measured S')
hold off