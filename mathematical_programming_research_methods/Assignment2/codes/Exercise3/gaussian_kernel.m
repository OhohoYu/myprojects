% function gaussian_kernel - Gaussian Kernel mapping. 
%
% Inputs - X1 and X2 are matrices in the input space i.e. matrices of features  
% computed from training/test sets. input c refers to the parameter c in
% K(p,q) = e^-c||p=q||^2.  
%
% Output - [kernel], Gaussian kernel mapping of the inputs. 
function [kernel] = gaussian_kernel(X1,X2, c)
    m = length(X1);
    n = length(X2);
    kernel = zeros(m,n);
    for i = 1:m % loop over X1 examples
        for j=1:n % loop over X2 examples
            % create kernel of dimensionality
            kernel(i,j) = exp(-(((norm(X1(i,:) - X2(j,:)))^2)*c));
        end
    end
end