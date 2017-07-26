% function my_pca.m performs PCA on a dataset. 
%
% Inputs: dataset X (mxn matrix), pc_no -> number of new principal
% components
%
% Output: Transformed coordinates using feature map
%
function transformed_data = my_pca(X, pc_no)
    % covariance matrix computations
    big_covar = X'*X;
    covar = big_covar/length(X);
    % apply singular value decomposition
    [u,~,~]=svd(covar);    
    % data transformation
    transformed_data = X*u(:,1:pc_no);
end