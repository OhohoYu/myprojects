clear 

% gradient descent function called
seq = graddesc('fc','dfc',[0,0],0.01,0.1);
x = seq(:,1); % x-axis values for points traversed
y = seq(:,2); % y-axis values for points traversed

plot(x,y) % 2D plotting function

