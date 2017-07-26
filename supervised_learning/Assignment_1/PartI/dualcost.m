% the following function is created using: 
% (1/l)(k_test*alpha - y)'(k_test*alpha - y)

function MSE=dualcost(k,y,alpha)
    [l,~]=size(y);
    MSE=1/l*(k*alpha-y)'*(k*alpha-y);                    
end

