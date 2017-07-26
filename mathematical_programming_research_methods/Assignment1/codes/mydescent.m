function soln = mydescent(A, b, guess, step, tol)
% gradient descent for linear regression
% A - matrix
% b - column vector
% guess - initial guess
% step - step size
% tol - tolerance
e = A * guess - b;    % initial error calculation
grad = [sum(e.*A(:,1)), sum(e.*A(:,2))];  % initial gradient calculation
seq = guess'; % sequence initialisation
while(norm(grad)>tol)  % crude termination condition for gradient descent.
    guess = guess - step*grad'; % coordinates update
    seq = [seq; guess']; % updates sequence of points
    e = A * guess - b; % error evaluation at t+1
    grad  = [sum(e.*A(:,1)), sum(e.*A(:,2))]; % gradient evaluation at t+1
end 
soln = seq; % matrix of final sequence of points
end



