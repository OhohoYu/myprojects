clear 

% matrix/vector declarations
A = [1, -1; 1, 1; 1, 2]; 
b = [1;1;3]; 

% initial guess= [0,0], step = 0.01, tol = 0.00001
seq = mydescent(A, b, [0;0],0.01,0.00001);
x1 = seq(:,1); % x-axis values for points traversed
x2 = seq(:,2); % y-axis values for points traversed

plot(x1,x2) % 2D plotting function

