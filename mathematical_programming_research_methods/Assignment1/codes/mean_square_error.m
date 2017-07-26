function MSE = mean_square_error(x, y)
% mean_square_error intakes two arrays x and y and calculates the MSE
% between them. both arrays can have any dimensions but must be of the same
% size and class. 
MSE = (norm(x(:)-y(:),2).^2)/numel(x);  % SSE/m

