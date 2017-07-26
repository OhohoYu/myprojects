% As part of the linear regression operation,intead of performing the
% calculation of the learned w each time, the follwoing fucntion is created
% which will be called.

% As the name of the fucntion suggests, w_calc calculates the learned w vector. 
% This is done by opertaing regression on the training se.

function fnc = w_calc(x,y)
fnc =mldivide((x.'*x),(x.'*y));
end

%  w = (X'X)^(-1) * (X'y)
%  where,
%  x is  the training/test set of x values.
%  y is the training/test set y-values.  
