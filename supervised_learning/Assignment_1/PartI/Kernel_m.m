function fcn = Kernel_m(x1,x2,v)
% Gaussian Ridge Regression Kernel
% function calculates gaussian kernel using Eq.11 from assignmnet handout
% Input:
% x1 training data (x1 and x2 should have same no of columns)
% x2 validation data
% sigma is the variance parameter of gaussian kernel
% Output:
% Kernel matrix (can b inspected by imagesc(K))

x = [x1; x2];
%xt=x';

[m, ~] = size(x);

fcn = zeros(m,m);

for i=1:m
    for j=1:m
       fcn(i,j) = exp(-(norm(x(i,:)-x(j,:),2).^2)/(2*v.^2));
    end
end
    