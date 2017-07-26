clear 

% matrix/vector declarations
A = [1, -1; 1, 1; 1, 2]; 
b = [1;1;3]; 

% initial guess= [0,0], step = 0.01, tol = 0.00001
mydescent(A, b, [0;0],0.01,0.00001)

