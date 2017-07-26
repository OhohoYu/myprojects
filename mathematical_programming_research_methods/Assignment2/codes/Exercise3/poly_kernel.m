% function poly_kernel - Polynomial Kernel mapping. 
%
% Inputs - X1 and X2 are matrices in the input space i.e. matrices of features  
% computed from training/test sets. c=0 in K(x,y) = (x'*y +c)^d so c is ignored.
% degree: the dimension of the polynomial.
%
% Output - [kernel], the polynomial kernel mapping of the inputs. 
function [kernel] = poly_kernel(X1,X2, degree)
    kernel = (X1*X2').^degree; 
end
