clear 

% gradient descent function called
seq = graddesc('fc','dfc',[0,0],0.01,0.1);
x = seq(:,1); % x-axis values for points traversed
y = seq(:,2); % y-axis values for points traversed
fxy = fcarg(x,y); % corresponding f(x,y) value

plot3(x,y,fxy); % 3D plotting function
grid on  % displays mayor grid lines in plot

% rotate graph to see trajectory better
