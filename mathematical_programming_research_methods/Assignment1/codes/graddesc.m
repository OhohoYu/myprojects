function soln = graddesc(f, g, i, e, tol)
% gradient descent
% f -- function
% g -- gradient
% i -- initial guess
% e -- step size
% tol -- tolerance
gi = feval(g,i) ; % evaluates initial gradient
seq = i; % sequence initialisation
while(norm(gi)>tol)  % crude termination condition
  i = i - e .* feval(g,i) ; % x(t+1) = x(t) - lambda*grad*f(x(t))
  gi = feval(g,i); % evaluates gradient at t+1
  seq = [seq;i]; % updates sequence of points
end
soln = seq; % matrix of final sequence of points (x - col 1, y - col 2)
end


