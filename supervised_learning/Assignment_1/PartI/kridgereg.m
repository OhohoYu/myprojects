% This function is programmed to perform kernel ridge regression 
% using alpha=y/(K+(gamma*l*I))        

function fcn = kridgereg(k,y,gamma)
% Number of training points:
l = size(y,1); 
% Kernel ridge regression:
fcn = mldivide((k + (gamma*l*eye(l))) , y);
end