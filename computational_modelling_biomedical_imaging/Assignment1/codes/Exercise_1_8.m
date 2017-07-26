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

rng(1)

% Define a starting point for the non-linear fit
startx = [3.5e+0   3e-03   2.5e-01 0 0];

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
% Now run the fitting
lb = [0; 0; 0; -inf; -inf];
ub = [inf; inf; 1; inf; inf];

% inverse transformation on starting point
newstartx(1) = sqrt(startx(1));
newstartx(2) = sqrt(startx(2));
newstartx(3) = -log((1/startx(3))-1);
newstartx(4) = startx(4);
newstartx(5) = startx(5);

t = cputime;
% Now rerun the fitting
[parameter_hat2,RESNORM2,EXITFLAG2,OUTPUT2]=fminunc('RicianSSDPart1_2',newstartx,...
                                                    h,Avox,bvals,qhat);
%[parameter_hat2,RESNORM2,EXITFLAG2,OUTPUT2]=fmincon('RicianSSDPart1_2',newstartx,...
%                                                    [],[],[],[],lb, ub,[], ...
%                                                    h,Avox,bvals,qhat);

e = cputime - t
                                                 
% average and standard deviation for predicted signal
[RESNORM_check2, S_pred2] = RicianSSDPart1_2(parameter_hat2, Avox, bvals, qhat);
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
ylim([0 5000])
legend('predicted S', 'measured S')
hold off