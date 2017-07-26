clear 
%
% Whac-A-Mole! Solver
%
% Intaking a matrix representation of a Whac-a-Mole board configuration, this
% program finds the sequence of holes that need to be hit in order to empty
% the board. This implementation makes use of Gaussian Elimination, which solves
% the puzzle but with the possibility of many redundant steps. 
%
% user specified number of rows/columns in the grid
n = 3;
grid = randi([0 1],n,n)
%
tog_rownum = n*n;
tog_mat = toggle_maker(n);
grid = grid';
grid = (reshape(grid, 1, tog_rownum))';
appended = [tog_mat, grid];

sol = solve_appended(appended)
result = sol(:,((n*n)+1));
result = (reshape(result, 3,3))'





