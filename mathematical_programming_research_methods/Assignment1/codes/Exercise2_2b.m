% part i

clear 

% function plot
X1 = linspace(0,1,100); % x-values for sin plot
Y1 = (sin(2*pi*X1)).^2;  % sin function
% noisy dataset plot
X2 = linspace(0,1,30); % sample uniformly 30 times from [0,1]
Y2 = gfunction(X2,0.07); % g_0.07 applied to each element of X2

plot(X1,Y1,'b') % sin function
ax = gca;
ax.XAxisLocation = 'origin';
hold on 
plot(X2,Y2,'or') % noisy dataset superimposed 
hold off

% part ii

clear

% for noisy dataset plot
X1 = linspace(0,1,30); % sample uniformly 30 times from [0,1]
Y1 = gfunction(X1,0.07); % g_0.07 applied to each element of X1

% coefficients returned for polynomial fits of noisy dataset 
P2 = polynomialfit(X1,Y1,2);
P5 = polynomialfit(X1,Y1,5);
P10 = polynomialfit(X1,Y1,10);
P14 = polynomialfit(X1,Y1,14);
P18 = polynomialfit(X1,Y1,18);

% for polynomial fits plot
XP = linspace(0,1,100);
Y2 = polynomialeval(P2,XP);
Y5 = polynomialeval(P5,XP);
Y10 = polynomialeval(P10,XP);
Y14 = polynomialeval(P14,XP);
Y18 = polynomialeval(P18,XP);

figure
plot(X1,Y1,'or') % noisy dataset plot 
ax = gca;
ax.XAxisLocation = 'origin';
hold on 
plot(XP,Y2,'--+c','LineWidth',0.4,'MarkerSize',2) % k=2 polynomial fit plot
plot(XP,Y5,'--b','LineWidth',2) % k=5
plot(XP,Y10, ':k','LineWidth',2) % k=10
plot(XP,Y14, '-.m','LineWidth',2) % k=14
plot(XP,Y18, '-g','LineWidth',1.5) % k=18
hold off

legend('S_{0.07,30}','k=2','k=5','k=10','k=14','k=18','Location','eastoutside')

