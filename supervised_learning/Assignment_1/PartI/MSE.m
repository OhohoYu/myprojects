% The Means Squared Error (MSE) is often used after regression operations,
% for training or test sets.
% The following fucntion is created which will be called.

function mse = MSE(x,y,w)

len = size(x,1);
for i = 1:len
   n_mse(i) = ((x(i,:)*w) - y(i,:)).^2;
end
mse = (1./len) .* sum(n_mse);

end
  

% MSE, = (1/number of rows) * sum((x*w-y)^2)
% where,
% x is the training/test data set x-values
% y is the training/test data set y-values
% w is a result of w_calc function 
 
