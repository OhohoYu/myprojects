function Y = polynomialeval(P,X) 
% if P is an array whose elements are the coefficients of a polynomial, 
% polynomialeval(P,X) is the value of the polynomial evaluated at X. 
    len = length(X);    % number of x-coords
    Y = zeros(1, len);  % number of y-coords
    for i = 1:len       % calculations for each point
        N = length(P) - 1;   % max polynomial degree
        e = N:-1:0;
        for n = 1:N+1   % calculations for each polynomial degree
            Y(i) = Y(i) + P(n) * X(i).^e(n); % evaluate polynomial recursively
        end
    end
    Y;
end



