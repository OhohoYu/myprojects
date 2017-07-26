function pfit = polynomialfit(X, Y, k)
% polynomialfit finds the coefficients of a polynomial p(X) of degree k-1 (basis
% dimension k) that fits the data
%
% X -> data point x-coords
% Y -> data point y-coords
% k - basis dimension
%
% construct Vandermonde matrix phi, a matrix of m rows and k columns, where
% phi(i,j) = phi_(k-j)(x_i) i.e. the last col is all ones (phi_1), the second last 
% column represents X_1, X_2,...X_m, etc. 
X = X(:);
phi = ones(length(X), k); % initialise Vandermonde matrix (m x k)
for j = k-1:-1:1  % update columns from last to first
   phi(:, j) = phi(:, j + 1) .* X; % recursively add a power of k 
end
% solve least squares problem computing the QR factorisation, which partitions the 
% Vandermonde matrix phi into Q and R. phi = Q*R where phi is mxn, Q is a unitary 
% (mxm) matrix and R is a mxn matrix.
[Q, R] = qr(phi);  
pfit = (R \ (Q' * Y(:)))'; 
end

