function fcn = w_ridge(x,y,gamma)
d = size(x,2);
l = size(x,1);
fcn = mldivide((x'*x)+(gamma*d*eye(d)) , (x'*y));
end