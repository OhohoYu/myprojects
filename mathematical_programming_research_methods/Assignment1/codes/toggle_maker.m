function [tog_mat] = toggle_maker(n)
% 
%   
tog_mat = zeros(n*n); % nrow/ncol of A = n*n 
  for i=1:n     % each M element has an associated n*n matrix
    for j=1:n 
      tog_row = n * (i - 1) + j; % row in A corresponding to each hole
      tog_mat(tog_row, tog_row) = 1; % whacked hole is toggled 
      % whacking hole not in top row toggles a hole below it
      if (i > 1)  
        tog_mat(tog_row, tog_row - n) = 1;
      end                  
      % whacking hole not in bottom row toggles a hole above it
      if (i < n)  
        tog_mat(tog_row, tog_row + n) = 1;
      end
      % whacking hole not in leftmost row toggles the hole to its left
      if (j > 1)  
        tog_mat(tog_row, tog_row - 1) = 1;
      end
      % whacking hole not in rightmost row toggles the hole to its right
      if (j < n)  
        tog_mat(tog_row, tog_row + 1) = 1;
      end
    end
  end
end

