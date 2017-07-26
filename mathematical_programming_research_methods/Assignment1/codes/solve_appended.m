function [AM] = solve_appended(AM)
% The function solve_appended solves the linear system of equations
% specified by the augmented matrix (A|M), where A is the toggle matrix of
% the grid and M is a column vector specifying the game's initial
% configuration. The function intakes the augmented matrix (A | M) and outputs 
% its row-reduced echelon form in gf(2). The output matrix corresponds to the 
% positions of the grid you have to whack to solve the problem
[m,n] = size(AM);
i = 1;
j = 1;
while (i <= m) && (j <= n) % Perform a while loop over the rows/cols of (A|M)
   % We will illustrate the steps for i=1, j=1.
   % 
   % First step:
   % We know that the element (1,1) is 1 from the configuration of our
   % toggle matrix. We proceed to find the second lowest entry (furthest up)
   % in the first column which is equal to 1. This will be s - our pivot. 
   s = i - 1 + find(AM(i:m,j),1);
   % Second step: We proceed to swap the position of the row corresponding to 
   % s - for our toggle matrix this is the 2nd row -  with the first row (i=1). 
   AM([i s],j:n) = AM([s i],j:n);
   % variable right_of_pivot is made up of the variables in the sth row to
   % the right of pivot s. 
   right_of_pivot = AM(i,j:n);
   % variable current_column consists of the jth column (in this case
   % column 1)
   current_column = AM(1:m,j);
   % our objective is to obtain a matrix "turn", that will indicate with ones, which  
   % entries of the matrix need to change status to become zeroes. we
   % obtain the matrix turn by multiplying current_column*right_of_pivot,
   % leaving all entries to the right of s in row s  equal to zero. also
   % leaving the entries of the jth column except that of s equal to zero.  
   % we will have to set the pivot to zero in this operation so that it is not 
   % flipped and is kept at one. (turn(1,1) will equal 1 and (1,1) is
   % switched from 0 to 1. 
   current_column(i) = 0; 
   turn = current_column*right_of_pivot;
   % XOR pivot RHS with all other rows
   AM(1:m,j:n) = xor( AM(1:m,j:n), turn);
   i = i + 1;  % repeat for all rows/columns
   j = j + 1;
end

% We will eventually arrive at an augmented matrix (A | M) where A is
% diagonal. M will indicate the value of its corresponding diagonal matrix,
% 1 or 0. 1 indicates that this hole needs to be whacked to empty the grid,
% 0 the contrary. 


